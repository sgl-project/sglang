// Validation implementation for Chat Completions API

use crate::protocols::common::StringOrArray;
use crate::protocols::openai::chat::request::ChatCompletionRequest;
use crate::protocols::openai::chat::types::{ChatMessage, ResponseFormat, UserMessageContent};
use crate::protocols::validation::{
    utils::{
        validate_common_request_params, validate_conflicting_parameters,
        validate_mutually_exclusive_options, validate_non_empty_array,
    },
    CompletionCountProvider, LogProbsProvider, SGLangExtensionsProvider, SamplingOptionsProvider,
    StopConditionsProvider, TokenLimitsProvider, ValidatableRequest, ValidationError,
};

impl SamplingOptionsProvider for ChatCompletionRequest {
    fn get_temperature(&self) -> Option<f32> {
        self.temperature
    }
    fn get_top_p(&self) -> Option<f32> {
        self.top_p
    }
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }
    fn get_presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }
}

impl StopConditionsProvider for ChatCompletionRequest {
    fn get_stop_sequences(&self) -> Option<&StringOrArray> {
        self.stop.as_ref()
    }
}

impl TokenLimitsProvider for ChatCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        // Prefer max_completion_tokens over max_tokens if both are set
        self.max_completion_tokens.or(self.max_tokens)
    }

    fn get_min_tokens(&self) -> Option<u32> {
        self.min_tokens
    }
}

impl LogProbsProvider for ChatCompletionRequest {
    fn get_logprobs(&self) -> Option<u32> {
        // For chat API, logprobs is a boolean, return 1 if true for validation purposes
        if self.logprobs {
            Some(1)
        } else {
            None
        }
    }

    fn get_top_logprobs(&self) -> Option<u32> {
        self.top_logprobs
    }
}

impl SGLangExtensionsProvider for ChatCompletionRequest {
    fn get_top_k(&self) -> Option<i32> {
        self.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.repetition_penalty
    }
}

impl CompletionCountProvider for ChatCompletionRequest {
    fn get_n(&self) -> Option<u32> {
        self.n
    }
}

impl ChatCompletionRequest {
    /// Validate message-specific requirements
    pub fn validate_messages(&self) -> Result<(), ValidationError> {
        // Ensure messages array is not empty
        validate_non_empty_array(&self.messages, "messages")?;

        // Validate message content is not empty
        for (i, msg) in self.messages.iter().enumerate() {
            if let ChatMessage::User { content, .. } = msg {
                match content {
                    UserMessageContent::Text(text) if text.is_empty() => {
                        return Err(ValidationError::InvalidValue {
                            parameter: format!("messages[{}].content", i),
                            value: "empty".to_string(),
                            reason: "message content cannot be empty".to_string(),
                        });
                    }
                    UserMessageContent::Parts(parts) if parts.is_empty() => {
                        return Err(ValidationError::InvalidValue {
                            parameter: format!("messages[{}].content", i),
                            value: "empty array".to_string(),
                            reason: "message content parts cannot be empty".to_string(),
                        });
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Validate response format if specified
    pub fn validate_response_format(&self) -> Result<(), ValidationError> {
        if let Some(ResponseFormat::JsonSchema { json_schema }) = &self.response_format {
            if json_schema.name.is_empty() {
                return Err(ValidationError::InvalidValue {
                    parameter: "response_format.json_schema.name".to_string(),
                    value: "empty".to_string(),
                    reason: "JSON schema name cannot be empty".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Validate chat API specific logprobs requirements
    pub fn validate_chat_logprobs(&self) -> Result<(), ValidationError> {
        // In chat API, if logprobs=true, top_logprobs must be specified
        if self.logprobs && self.top_logprobs.is_none() {
            return Err(ValidationError::MissingRequired {
                parameter: "top_logprobs".to_string(),
            });
        }

        // If top_logprobs is specified, logprobs should be true
        if self.top_logprobs.is_some() && !self.logprobs {
            return Err(ValidationError::InvalidValue {
                parameter: "logprobs".to_string(),
                value: "false".to_string(),
                reason: "must be true when top_logprobs is specified".to_string(),
            });
        }

        Ok(())
    }

    /// Validate cross-parameter relationships specific to chat completions
    pub fn validate_chat_cross_parameters(&self) -> Result<(), ValidationError> {
        // Validate that both max_tokens and max_completion_tokens aren't set
        validate_conflicting_parameters(
            "max_tokens",
            self.max_tokens.is_some(),
            "max_completion_tokens",
            self.max_completion_tokens.is_some(),
            "cannot specify both max_tokens and max_completion_tokens",
        )?;

        // Validate that tools and functions aren't both specified (deprecated)
        validate_conflicting_parameters(
            "tools",
            self.tools.is_some(),
            "functions",
            self.functions.is_some(),
            "functions is deprecated, use tools instead",
        )?;

        // Validate structured output constraints don't conflict with JSON response format
        let has_json_format = matches!(
            self.response_format,
            Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
        );

        validate_conflicting_parameters(
            "response_format",
            has_json_format,
            "regex",
            self.regex.is_some(),
            "cannot use regex constraint with JSON response format",
        )?;

        validate_conflicting_parameters(
            "response_format",
            has_json_format,
            "ebnf",
            self.ebnf.is_some(),
            "cannot use EBNF constraint with JSON response format",
        )?;

        // Only one structured output constraint should be active
        let structured_constraints = [
            ("regex", self.regex.is_some()),
            ("ebnf", self.ebnf.is_some()),
            (
                "json_schema",
                matches!(
                    self.response_format,
                    Some(ResponseFormat::JsonSchema { .. })
                ),
            ),
        ];

        validate_mutually_exclusive_options(
            &structured_constraints,
            "Only one structured output constraint (regex, ebnf, or json_schema) can be active at a time",
        )?;

        Ok(())
    }
}

impl ValidatableRequest for ChatCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        // Call the common validation function from the validation module
        validate_common_request_params(self)?;

        // Then validate chat-specific parameters
        self.validate_messages()?;
        self.validate_response_format()?;
        self.validate_chat_logprobs()?;
        self.validate_chat_cross_parameters()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::chat::types::*;

    fn create_valid_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Text("Hello".to_string()),
                name: None,
            }],
            temperature: Some(1.0),
            top_p: Some(0.9),
            n: Some(1),
            stream: false,
            stream_options: None,
            stop: None,
            max_tokens: Some(100),
            max_completion_tokens: None,
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            logit_bias: None,
            user: None,
            seed: None,
            logprobs: false,
            top_logprobs: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            functions: None,
            function_call: None,
            // SGLang extensions
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
            lora_path: None,
            session_params: None,
            separate_reasoning: true,
            stream_reasoning: true,
            return_hidden_states: false,
        }
    }

    #[test]
    fn test_valid_chat_request() {
        let request = create_valid_request();
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_invalid_temperature() {
        let mut request = create_valid_request();
        request.temperature = Some(3.0); // Too high

        let result = request.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            ValidationError::OutOfRange { parameter, .. } => {
                assert_eq!(parameter, "temperature");
            }
            _ => panic!("Expected OutOfRange error"),
        }
    }

    #[test]
    fn test_invalid_top_p() {
        let mut request = create_valid_request();
        request.top_p = Some(1.5); // Too high

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_too_many_stop_sequences() {
        let mut request = create_valid_request();
        request.stop = Some(StringOrArray::Array(vec![
            "stop1".to_string(),
            "stop2".to_string(),
            "stop3".to_string(),
            "stop4".to_string(),
            "stop5".to_string(), // Too many
        ]));

        let result = request.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_stop_sequence() {
        let mut request = create_valid_request();
        request.stop = Some(StringOrArray::String("".to_string()));

        let result = request.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            ValidationError::InvalidValue {
                parameter, reason, ..
            } => {
                assert_eq!(parameter, "stop");
                assert!(reason.contains("empty"));
            }
            _ => panic!("Expected InvalidValue error"),
        }
    }

    #[test]
    fn test_empty_messages() {
        let mut request = create_valid_request();
        request.messages = vec![];

        let result = request.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            ValidationError::MissingRequired { parameter } => {
                assert_eq!(parameter, "messages");
            }
            _ => panic!("Expected MissingRequired error"),
        }
    }

    #[test]
    fn test_invalid_n_parameter() {
        let mut request = create_valid_request();
        request.n = Some(0);

        let result = request.validate();
        assert!(result.is_err());

        request.n = Some(20); // Too high
        assert!(request.validate().is_err());
    }

    #[test]
    fn test_conflicting_max_tokens() {
        let mut request = create_valid_request();
        request.max_tokens = Some(100);
        request.max_completion_tokens = Some(200);

        let result = request.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            ValidationError::ConflictingParameters {
                parameter1,
                parameter2,
                ..
            } => {
                assert!(parameter1.contains("max_tokens"));
                assert!(parameter2.contains("max_completion_tokens"));
            }
            _ => panic!("Expected ConflictingParameters error"),
        }
    }

    #[test]
    fn test_logprobs_without_top_logprobs() {
        let mut request = create_valid_request();
        request.logprobs = true;
        request.top_logprobs = None;

        let result = request.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_sglang_extensions() {
        let mut request = create_valid_request();

        // Valid top_k
        request.top_k = Some(-1); // Disabled
        assert!(request.validate().is_ok());

        request.top_k = Some(50); // Valid positive
        assert!(request.validate().is_ok());

        request.top_k = Some(0); // Invalid
        assert!(request.validate().is_err());

        // Valid min_p
        request.top_k = None;
        request.min_p = Some(0.1);
        assert!(request.validate().is_ok());

        request.min_p = Some(1.5); // Too high
        assert!(request.validate().is_err());

        // Valid repetition_penalty
        request.min_p = None;
        request.repetition_penalty = Some(1.2);
        assert!(request.validate().is_ok());

        request.repetition_penalty = Some(0.0); // Valid - minimum value
        assert!(request.validate().is_ok());

        request.repetition_penalty = Some(2.0); // Valid - maximum value
        assert!(request.validate().is_ok());

        request.repetition_penalty = Some(2.1); // Invalid - too high
        assert!(request.validate().is_err());

        request.repetition_penalty = Some(-0.1); // Invalid - negative
        assert!(request.validate().is_err());
    }

    #[test]
    fn test_structured_output_conflicts() {
        let mut request = create_valid_request();

        // JSON response format with regex should conflict
        request.response_format = Some(ResponseFormat::JsonObject);
        request.regex = Some(".*".to_string());

        let result = request.validate();
        assert!(result.is_err());

        // Multiple structured constraints should conflict
        request.response_format = None;
        request.regex = Some(".*".to_string());
        request.ebnf = Some("grammar".to_string());

        let result = request.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_min_max_tokens_validation() {
        let mut request = create_valid_request();
        request.min_tokens = Some(100);
        request.max_tokens = Some(50); // min > max

        let result = request.validate();
        assert!(result.is_err());

        // Should work with max_completion_tokens too
        request.max_tokens = None;
        request.max_completion_tokens = Some(200);
        request.min_tokens = Some(100);
        assert!(request.validate().is_ok());
    }
}
