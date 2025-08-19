// Validation implementation for Chat Completions API

use crate::protocols::validation::{
    constants::*, 
    utils::*,
    ValidationError,
    SamplingOptionsProvider,
    StopConditionsProvider, 
    TokenLimitsProvider,
    LogProbsProvider,
    ValidatableRequest,
};
use crate::protocols::openai::chat::request::ChatCompletionRequest;
use crate::protocols::openai::chat::types::{ChatMessage, UserMessageContent, ResponseFormat};
use crate::protocols::common::StringOrArray;

impl SamplingOptionsProvider for ChatCompletionRequest {
    fn validate_sampling_options(&self) -> Result<(), ValidationError> {
        // Validate temperature (0.0 to 2.0)
        if let Some(temp) = self.temperature {
            validate_range(temp, &TEMPERATURE_RANGE, "temperature")?;
        }
        
        // Validate top_p (0.0 to 1.0)
        if let Some(top_p) = self.top_p {
            validate_range(top_p, &TOP_P_RANGE, "top_p")?;
        }
        
        // Validate frequency_penalty (-2.0 to 2.0)
        if let Some(freq_penalty) = self.frequency_penalty {
            validate_range(freq_penalty, &FREQUENCY_PENALTY_RANGE, "frequency_penalty")?;
        }
        
        // Validate presence_penalty (-2.0 to 2.0)
        if let Some(pres_penalty) = self.presence_penalty {
            validate_range(pres_penalty, &PRESENCE_PENALTY_RANGE, "presence_penalty")?;
        }
        
        Ok(())
    }
    
    fn get_temperature(&self) -> Option<f32> { self.temperature }
    fn get_top_p(&self) -> Option<f32> { self.top_p }
    fn get_frequency_penalty(&self) -> Option<f32> { self.frequency_penalty }
    fn get_presence_penalty(&self) -> Option<f32> { self.presence_penalty }
}

impl StopConditionsProvider for ChatCompletionRequest {
    fn validate_stop_conditions(&self) -> Result<(), ValidationError> {
        // Validate stop sequences (max 4)
        if let Some(ref stop) = self.stop {
            validate_max_items(&stop.to_vec(), MAX_STOP_SEQUENCES, "stop")?;
            
            // Ensure no empty stop sequences
            match stop {
                StringOrArray::String(s) if s.is_empty() => {
                    return Err(ValidationError::InvalidValue {
                        parameter: "stop".to_string(),
                        value: "empty string".to_string(),
                        reason: "stop sequences cannot be empty".to_string(),
                    });
                }
                StringOrArray::Array(arr) => {
                    for (i, s) in arr.iter().enumerate() {
                        if s.is_empty() {
                            return Err(ValidationError::InvalidValue {
                                parameter: format!("stop[{}]", i),
                                value: "empty string".to_string(),
                                reason: "stop sequences cannot be empty".to_string(),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    fn get_stop_sequences(&self) -> Option<&StringOrArray> {
        self.stop.as_ref()
    }
}

impl TokenLimitsProvider for ChatCompletionRequest {
    fn validate_token_limits(&self) -> Result<(), ValidationError> {
        // Validate max_tokens if provided
        if let Some(max_tokens) = self.max_tokens {
            validate_positive(max_tokens, "max_tokens")?;
        }
        
        // Validate max_completion_tokens if provided (newer parameter)
        if let Some(max_completion_tokens) = self.max_completion_tokens {
            validate_positive(max_completion_tokens, "max_completion_tokens")?;
        }
        
        // SGLang extension: validate min_tokens
        if let Some(min_tokens) = self.min_tokens {
            validate_positive(min_tokens, "min_tokens")?;
        }
        
        Ok(())
    }
    
    fn get_max_tokens(&self) -> Option<u32> { 
        // Prefer max_completion_tokens over max_tokens if both are set
        self.max_completion_tokens.or(self.max_tokens)
    }
    
    fn get_min_tokens(&self) -> Option<u32> { self.min_tokens }
}

impl LogProbsProvider for ChatCompletionRequest {
    fn validate_logprobs(&self) -> Result<(), ValidationError> {
        // Validate top_logprobs (0 to 20 for chat API)
        if let Some(top_logprobs) = self.top_logprobs {
            validate_range(top_logprobs, &TOP_LOGPROBS_RANGE, "top_logprobs")?;
        }
        
        // If logprobs is true, top_logprobs should be set
        if self.logprobs && self.top_logprobs.is_none() {
            return Err(ValidationError::InvalidValue {
                parameter: "logprobs".to_string(),
                value: "true".to_string(),
                reason: "when logprobs is true, top_logprobs must be specified".to_string(),
            });
        }
        
        Ok(())
    }
    
    fn get_logprobs(&self) -> Option<u32> { 
        // For chat API, logprobs is a boolean, return 1 if true
        if self.logprobs { Some(1) } else { None }
    }
    
    fn get_top_logprobs(&self) -> Option<u32> { self.top_logprobs }
}

impl ChatCompletionRequest {
    /// Validate SGLang-specific extensions
    pub fn validate_sglang_extensions(&self) -> Result<(), ValidationError> {
        // Validate top_k (-1 to disable, or positive)
        if let Some(top_k) = self.top_k {
            validate_top_k(top_k)?;
        }
        
        // Validate min_p (0.0 to 1.0)
        if let Some(min_p) = self.min_p {
            validate_range(min_p, &sglang::MIN_P_RANGE, "min_p")?;
        }
        
        // Validate repetition_penalty (must be positive)
        if let Some(rep_penalty) = self.repetition_penalty {
            if rep_penalty <= sglang::REPETITION_PENALTY_MIN {
                return Err(ValidationError::InvalidValue {
                    parameter: "repetition_penalty".to_string(),
                    value: rep_penalty.to_string(),
                    reason: "must be positive".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate message-specific requirements
    pub fn validate_messages(&self) -> Result<(), ValidationError> {
        // Ensure messages array is not empty
        if self.messages.is_empty() {
            return Err(ValidationError::MissingRequired {
                parameter: "messages".to_string(),
            });
        }
        
        // Validate message sequence (optional but recommended)
        // System messages should come first, followed by user/assistant alternation
        let mut _last_role = None;
        for (i, msg) in self.messages.iter().enumerate() {
            match msg {
                ChatMessage::System { .. } => {
                    // System messages should typically be at the beginning
                    if i > 0 && !matches!(self.messages[i-1], ChatMessage::System { .. }) {
                        // This is a warning, not an error - some use cases may need this
                    }
                }
                ChatMessage::User { content, .. } => {
                    // Validate user message content is not empty
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
                _ => {}
            }
            
            _last_role = Some(msg);
        }
        
        Ok(())
    }
    
    /// Validate response format if specified
    pub fn validate_response_format(&self) -> Result<(), ValidationError> {
        if let Some(ref format) = self.response_format {
            match format {
                ResponseFormat::JsonObject => {
                    // JSON mode requires compatible model
                    // This is a runtime check, not a static validation
                }
                ResponseFormat::JsonSchema { json_schema } => {
                    // Validate JSON schema has required fields
                    if json_schema.name.is_empty() {
                        return Err(ValidationError::InvalidValue {
                            parameter: "response_format.json_schema.name".to_string(),
                            value: "empty".to_string(),
                            reason: "JSON schema name cannot be empty".to_string(),
                        });
                    }
                    
                    // Schema should be a valid JSON schema object
                    // This would require deeper validation in production
                }
                ResponseFormat::Text => {
                    // Text format has no special requirements
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate n parameter
    pub fn validate_n_parameter(&self) -> Result<(), ValidationError> {
        if let Some(n) = self.n {
            if n == 0 {
                return Err(ValidationError::InvalidValue {
                    parameter: "n".to_string(),
                    value: "0".to_string(),
                    reason: "must be at least 1".to_string(),
                });
            }
            
            // Most providers limit n to a reasonable number (e.g., 10)
            const MAX_N: u32 = 10;
            if n > MAX_N {
                return Err(ValidationError::InvalidValue {
                    parameter: "n".to_string(),
                    value: n.to_string(),
                    reason: format!("cannot exceed {}", MAX_N),
                });
            }
        }
        
        Ok(())
    }
}

impl ValidatableRequest for ChatCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        // Validate all standard parameters
        self.validate_sampling_options()?;
        self.validate_stop_conditions()?;
        self.validate_token_limits()?;
        self.validate_logprobs()?;
        
        // Validate chat-specific parameters
        self.validate_messages()?;
        self.validate_response_format()?;
        self.validate_n_parameter()?;
        
        // Validate SGLang extensions
        self.validate_sglang_extensions()?;
        
        // Cross-parameter validation
        self.validate_cross_parameters()?;
        
        Ok(())
    }
    
    fn validate_cross_parameters(&self) -> Result<(), ValidationError> {
        // Check min_tokens <= max_tokens if both are specified
        if let (Some(min_tokens), Some(max_tokens)) = (self.min_tokens, self.get_max_tokens()) {
            if min_tokens > max_tokens {
                return Err(ValidationError::ConflictingParameters {
                    parameter1: "min_tokens".to_string(),
                    parameter2: "max_tokens/max_completion_tokens".to_string(),
                    reason: format!("min_tokens ({}) cannot be greater than max_tokens ({})", min_tokens, max_tokens),
                });
            }
        }
        
        // Warn about conflicting parameters (temperature vs deterministic settings)
        if let Some(temp) = self.temperature {
            if temp == 0.0 && self.top_p.is_some() && self.top_p != Some(1.0) {
                // This is a warning - temperature=0 makes sampling deterministic,
                // so top_p has no effect
            }
        }
        
        // Validate that both max_tokens and max_completion_tokens aren't set
        if self.max_tokens.is_some() && self.max_completion_tokens.is_some() {
            return Err(ValidationError::ConflictingParameters {
                parameter1: "max_tokens".to_string(),
                parameter2: "max_completion_tokens".to_string(),
                reason: "cannot specify both max_tokens and max_completion_tokens".to_string(),
            });
        }
        
        // Validate that tools and functions aren't both specified (deprecated)
        if self.tools.is_some() && self.functions.is_some() {
            return Err(ValidationError::ConflictingParameters {
                parameter1: "tools".to_string(),
                parameter2: "functions".to_string(),
                reason: "functions is deprecated, use tools instead".to_string(),
            });
        }
        
        // Validate structured output constraints don't conflict
        if let Some(ref response_format) = self.response_format {
            // Can't have both response_format JSON and regex/ebnf constraints
            if matches!(response_format, ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. }) {
                if self.regex.is_some() {
                    return Err(ValidationError::ConflictingParameters {
                        parameter1: "response_format".to_string(),
                        parameter2: "regex".to_string(),
                        reason: "cannot use regex constraint with JSON response format".to_string(),
                    });
                }
                if self.ebnf.is_some() {
                    return Err(ValidationError::ConflictingParameters {
                        parameter1: "response_format".to_string(),
                        parameter2: "ebnf".to_string(),
                        reason: "cannot use EBNF constraint with JSON response format".to_string(),
                    });
                }
            }
        }
        
        // Only one structured output constraint should be active
        let structured_constraints = [
            self.regex.is_some(),
            self.ebnf.is_some(),
            matches!(self.response_format, Some(ResponseFormat::JsonSchema { .. })),
        ];
        let active_count = structured_constraints.iter().filter(|&&x| x).count();
        if active_count > 1 {
            return Err(ValidationError::Custom(
                "Only one structured output constraint (regex, ebnf, or json_schema) can be active at a time".to_string()
            ));
        }
        
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
            messages: vec![
                ChatMessage::User {
                    role: "user".to_string(),
                    content: UserMessageContent::Text("Hello".to_string()),
                    name: None,
                },
            ],
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
            ValidationError::InvalidValue { parameter, reason, .. } => {
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
            ValidationError::ConflictingParameters { parameter1, parameter2, .. } => {
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
        
        request.repetition_penalty = Some(0.0); // Invalid
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