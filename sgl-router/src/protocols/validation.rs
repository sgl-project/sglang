// Core validation infrastructure for API parameter validation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

// Import types from spec module
use crate::protocols::spec::{
    ChatCompletionRequest, ChatMessage, ResponseFormat, StringOrArray, UserMessageContent,
};

/// Validation constants for OpenAI API parameters
pub mod constants {
    /// Temperature range: 0.0 to 2.0 (OpenAI spec)
    pub const TEMPERATURE_RANGE: (f32, f32) = (0.0, 2.0);

    /// Top-p range: 0.0 to 1.0 (exclusive of 0.0)
    pub const TOP_P_RANGE: (f32, f32) = (0.0, 1.0);

    /// Presence penalty range: -2.0 to 2.0 (OpenAI spec)
    pub const PRESENCE_PENALTY_RANGE: (f32, f32) = (-2.0, 2.0);

    /// Frequency penalty range: -2.0 to 2.0 (OpenAI spec)
    pub const FREQUENCY_PENALTY_RANGE: (f32, f32) = (-2.0, 2.0);

    /// Logprobs range for completions API: 0 to 5
    pub const LOGPROBS_RANGE: (u32, u32) = (0, 5);

    /// Top logprobs range for chat completions: 0 to 20
    pub const TOP_LOGPROBS_RANGE: (u32, u32) = (0, 20);

    /// Maximum number of stop sequences allowed
    pub const MAX_STOP_SEQUENCES: usize = 4;

    /// SGLang-specific validation constants
    pub mod sglang {
        /// Min-p range: 0.0 to 1.0 (SGLang extension)
        pub const MIN_P_RANGE: (f32, f32) = (0.0, 1.0);

        /// Top-k minimum value: -1 to disable, otherwise positive
        pub const TOP_K_MIN: i32 = -1;

        /// Repetition penalty range: 0.0 to 2.0 (SGLang extension)
        /// 1.0 = no penalty, >1.0 = discourage repetition, <1.0 = encourage repetition
        pub const REPETITION_PENALTY_RANGE: (f32, f32) = (0.0, 2.0);
    }
}

/// Core validation error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Parameter value out of valid range
    OutOfRange {
        parameter: String,
        value: String,
        min: String,
        max: String,
    },
    /// Invalid parameter value format or type
    InvalidValue {
        parameter: String,
        value: String,
        reason: String,
    },
    /// Cross-parameter validation failure
    ConflictingParameters {
        parameter1: String,
        parameter2: String,
        reason: String,
    },
    /// Required parameter missing
    MissingRequired { parameter: String },
    /// Too many items in array parameter
    TooManyItems {
        parameter: String,
        count: usize,
        max: usize,
    },
    /// Custom validation error
    Custom(String),
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::OutOfRange {
                parameter,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "Parameter '{}' must be between {} and {}, got {}",
                    parameter, min, max, value
                )
            }
            ValidationError::InvalidValue {
                parameter,
                value,
                reason,
            } => {
                write!(
                    f,
                    "Invalid value for parameter '{}': {} ({})",
                    parameter, value, reason
                )
            }
            ValidationError::ConflictingParameters {
                parameter1,
                parameter2,
                reason,
            } => {
                write!(
                    f,
                    "Conflicting parameters '{}' and '{}': {}",
                    parameter1, parameter2, reason
                )
            }
            ValidationError::MissingRequired { parameter } => {
                write!(f, "Required parameter '{}' is missing", parameter)
            }
            ValidationError::TooManyItems {
                parameter,
                count,
                max,
            } => {
                write!(
                    f,
                    "Parameter '{}' has too many items: {} (maximum: {})",
                    parameter, count, max
                )
            }
            ValidationError::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Core validation utility functions
pub mod utils {
    use super::*;

    /// Validate that a numeric value is within the specified range (inclusive)
    pub fn validate_range<T>(
        value: T,
        range: &(T, T),
        param_name: &str,
    ) -> Result<T, ValidationError>
    where
        T: PartialOrd + Display + Copy,
    {
        if value >= range.0 && value <= range.1 {
            Ok(value)
        } else {
            Err(ValidationError::OutOfRange {
                parameter: param_name.to_string(),
                value: value.to_string(),
                min: range.0.to_string(),
                max: range.1.to_string(),
            })
        }
    }

    /// Validate that a positive number is actually positive
    pub fn validate_positive<T>(value: T, param_name: &str) -> Result<T, ValidationError>
    where
        T: PartialOrd + Display + Copy + Default,
    {
        if value > T::default() {
            Ok(value)
        } else {
            Err(ValidationError::InvalidValue {
                parameter: param_name.to_string(),
                value: value.to_string(),
                reason: "must be positive".to_string(),
            })
        }
    }

    /// Validate that an array doesn't exceed maximum length
    pub fn validate_max_items<T>(
        items: &[T],
        max_count: usize,
        param_name: &str,
    ) -> Result<(), ValidationError> {
        if items.len() <= max_count {
            Ok(())
        } else {
            Err(ValidationError::TooManyItems {
                parameter: param_name.to_string(),
                count: items.len(),
                max: max_count,
            })
        }
    }

    /// Validate that a required parameter is present
    pub fn validate_required<'a, T>(
        value: &'a Option<T>,
        param_name: &str,
    ) -> Result<&'a T, ValidationError> {
        value
            .as_ref()
            .ok_or_else(|| ValidationError::MissingRequired {
                parameter: param_name.to_string(),
            })
    }

    /// Validate top_k parameter (SGLang extension)
    pub fn validate_top_k(top_k: i32) -> Result<i32, ValidationError> {
        if top_k == constants::sglang::TOP_K_MIN || top_k > 0 {
            Ok(top_k)
        } else {
            Err(ValidationError::InvalidValue {
                parameter: "top_k".to_string(),
                value: top_k.to_string(),
                reason: "must be -1 (disabled) or positive".to_string(),
            })
        }
    }

    /// Generic validation function for sampling options
    pub fn validate_sampling_options<T: SamplingOptionsProvider + ?Sized>(
        request: &T,
    ) -> Result<(), ValidationError> {
        // Validate temperature (0.0 to 2.0)
        if let Some(temp) = request.get_temperature() {
            validate_range(temp, &constants::TEMPERATURE_RANGE, "temperature")?;
        }

        // Validate top_p (0.0 to 1.0)
        if let Some(top_p) = request.get_top_p() {
            validate_range(top_p, &constants::TOP_P_RANGE, "top_p")?;
        }

        // Validate frequency_penalty (-2.0 to 2.0)
        if let Some(freq_penalty) = request.get_frequency_penalty() {
            validate_range(
                freq_penalty,
                &constants::FREQUENCY_PENALTY_RANGE,
                "frequency_penalty",
            )?;
        }

        // Validate presence_penalty (-2.0 to 2.0)
        if let Some(pres_penalty) = request.get_presence_penalty() {
            validate_range(
                pres_penalty,
                &constants::PRESENCE_PENALTY_RANGE,
                "presence_penalty",
            )?;
        }

        Ok(())
    }

    /// Generic validation function for stop conditions
    pub fn validate_stop_conditions<T: StopConditionsProvider + ?Sized>(
        request: &T,
    ) -> Result<(), ValidationError> {
        if let Some(stop) = request.get_stop_sequences() {
            match stop {
                StringOrArray::String(s) => {
                    if s.is_empty() {
                        return Err(ValidationError::InvalidValue {
                            parameter: "stop".to_string(),
                            value: "empty string".to_string(),
                            reason: "stop sequences cannot be empty".to_string(),
                        });
                    }
                }
                StringOrArray::Array(arr) => {
                    validate_max_items(arr, constants::MAX_STOP_SEQUENCES, "stop")?;
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
            }
        }

        Ok(())
    }

    /// Generic validation function for token limits
    pub fn validate_token_limits<T: TokenLimitsProvider + ?Sized>(
        request: &T,
    ) -> Result<(), ValidationError> {
        // Validate max_tokens if provided
        if let Some(max_tokens) = request.get_max_tokens() {
            validate_positive(max_tokens, "max_tokens")?;
        }

        // Validate min_tokens if provided (SGLang extension)
        if let Some(min_tokens) = request.get_min_tokens() {
            validate_positive(min_tokens, "min_tokens")?;
        }

        Ok(())
    }

    /// Generic validation function for logprobs
    pub fn validate_logprobs<T: LogProbsProvider + ?Sized>(
        request: &T,
    ) -> Result<(), ValidationError> {
        // Validate logprobs (completions API - 0 to 5)
        if let Some(logprobs) = request.get_logprobs() {
            validate_range(logprobs, &constants::LOGPROBS_RANGE, "logprobs")?;
        }

        // Validate top_logprobs (chat API - 0 to 20)
        if let Some(top_logprobs) = request.get_top_logprobs() {
            validate_range(top_logprobs, &constants::TOP_LOGPROBS_RANGE, "top_logprobs")?;
        }

        Ok(())
    }

    /// Generic cross-parameter validation
    pub fn validate_cross_parameters<T: TokenLimitsProvider + ?Sized>(
        request: &T,
    ) -> Result<(), ValidationError> {
        // Check min_tokens <= max_tokens if both are specified
        if let (Some(min_tokens), Some(max_tokens)) =
            (request.get_min_tokens(), request.get_max_tokens())
        {
            if min_tokens > max_tokens {
                return Err(ValidationError::ConflictingParameters {
                    parameter1: "min_tokens".to_string(),
                    parameter2: "max_tokens".to_string(),
                    reason: format!(
                        "min_tokens ({}) cannot be greater than max_tokens ({})",
                        min_tokens, max_tokens
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validate conflicting structured output constraints
    pub fn validate_conflicting_parameters(
        param1_name: &str,
        param1_value: bool,
        param2_name: &str,
        param2_value: bool,
        reason: &str,
    ) -> Result<(), ValidationError> {
        if param1_value && param2_value {
            return Err(ValidationError::ConflictingParameters {
                parameter1: param1_name.to_string(),
                parameter2: param2_name.to_string(),
                reason: reason.to_string(),
            });
        }
        Ok(())
    }

    /// Validate that only one option from a set is active
    pub fn validate_mutually_exclusive_options(
        options: &[(&str, bool)],
        error_msg: &str,
    ) -> Result<(), ValidationError> {
        let active_count = options.iter().filter(|(_, is_active)| *is_active).count();
        if active_count > 1 {
            return Err(ValidationError::Custom(error_msg.to_string()));
        }
        Ok(())
    }

    /// Generic validation for SGLang extensions
    pub fn validate_sglang_extensions<T: SGLangExtensionsProvider + ?Sized>(
        request: &T,
    ) -> Result<(), ValidationError> {
        // Validate top_k (-1 to disable, or positive)
        if let Some(top_k) = request.get_top_k() {
            validate_top_k(top_k)?;
        }

        // Validate min_p (0.0 to 1.0)
        if let Some(min_p) = request.get_min_p() {
            validate_range(min_p, &constants::sglang::MIN_P_RANGE, "min_p")?;
        }

        // Validate repetition_penalty (0.0 to 2.0)
        if let Some(rep_penalty) = request.get_repetition_penalty() {
            validate_range(
                rep_penalty,
                &constants::sglang::REPETITION_PENALTY_RANGE,
                "repetition_penalty",
            )?;
        }

        Ok(())
    }

    /// Generic validation for n parameter (number of completions)
    pub fn validate_completion_count<T: CompletionCountProvider + ?Sized>(
        request: &T,
    ) -> Result<(), ValidationError> {
        const N_RANGE: (u32, u32) = (1, 10);

        if let Some(n) = request.get_n() {
            validate_range(n, &N_RANGE, "n")?;
        }

        Ok(())
    }

    /// Validate that an array is not empty
    pub fn validate_non_empty_array<T>(
        items: &[T],
        param_name: &str,
    ) -> Result<(), ValidationError> {
        if items.is_empty() {
            return Err(ValidationError::MissingRequired {
                parameter: param_name.to_string(),
            });
        }
        Ok(())
    }

    /// Validate common request parameters that are shared across all API types
    pub fn validate_common_request_params<T>(request: &T) -> Result<(), ValidationError>
    where
        T: SamplingOptionsProvider
            + StopConditionsProvider
            + TokenLimitsProvider
            + LogProbsProvider
            + SGLangExtensionsProvider
            + CompletionCountProvider
            + ?Sized,
    {
        // Validate all standard parameters
        validate_sampling_options(request)?;
        validate_stop_conditions(request)?;
        validate_token_limits(request)?;
        validate_logprobs(request)?;

        // Validate SGLang extensions and completion count
        validate_sglang_extensions(request)?;
        validate_completion_count(request)?;

        // Perform cross-parameter validation
        validate_cross_parameters(request)?;

        Ok(())
    }
}

/// Core validation traits for different parameter categories
pub trait SamplingOptionsProvider {
    /// Get temperature parameter
    fn get_temperature(&self) -> Option<f32>;

    /// Get top_p parameter
    fn get_top_p(&self) -> Option<f32>;

    /// Get frequency penalty parameter
    fn get_frequency_penalty(&self) -> Option<f32>;

    /// Get presence penalty parameter
    fn get_presence_penalty(&self) -> Option<f32>;
}

/// Trait for validating stop conditions
pub trait StopConditionsProvider {
    /// Get stop sequences
    fn get_stop_sequences(&self) -> Option<&StringOrArray>;
}

/// Trait for validating token limits
pub trait TokenLimitsProvider {
    /// Get maximum tokens parameter
    fn get_max_tokens(&self) -> Option<u32>;

    /// Get minimum tokens parameter (SGLang extension)
    fn get_min_tokens(&self) -> Option<u32>;
}

/// Trait for validating logprobs parameters
pub trait LogProbsProvider {
    /// Get logprobs parameter (completions API)
    fn get_logprobs(&self) -> Option<u32>;

    /// Get top_logprobs parameter (chat API)
    fn get_top_logprobs(&self) -> Option<u32>;
}

/// Trait for SGLang-specific extensions
pub trait SGLangExtensionsProvider {
    /// Get top_k parameter
    fn get_top_k(&self) -> Option<i32> {
        None
    }

    /// Get min_p parameter
    fn get_min_p(&self) -> Option<f32> {
        None
    }

    /// Get repetition_penalty parameter
    fn get_repetition_penalty(&self) -> Option<f32> {
        None
    }
}

/// Trait for n parameter (number of completions)
pub trait CompletionCountProvider {
    /// Get n parameter
    fn get_n(&self) -> Option<u32> {
        None
    }
}

/// Comprehensive validation trait that combines all validation aspects
pub trait ValidatableRequest:
    SamplingOptionsProvider
    + StopConditionsProvider
    + TokenLimitsProvider
    + LogProbsProvider
    + SGLangExtensionsProvider
    + CompletionCountProvider
{
    /// Perform comprehensive validation of the entire request
    fn validate(&self) -> Result<(), ValidationError> {
        // Use the common validation function
        utils::validate_common_request_params(self)
    }
}

// ==================================================================
// =            OPENAI CHAT COMPLETION VALIDATION                   =
// ==================================================================

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
        utils::validate_non_empty_array(&self.messages, "messages")?;

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
        utils::validate_conflicting_parameters(
            "max_tokens",
            self.max_tokens.is_some(),
            "max_completion_tokens",
            self.max_completion_tokens.is_some(),
            "cannot specify both max_tokens and max_completion_tokens",
        )?;

        // Validate that tools and functions aren't both specified (deprecated)
        utils::validate_conflicting_parameters(
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

        utils::validate_conflicting_parameters(
            "response_format",
            has_json_format,
            "regex",
            self.regex.is_some(),
            "cannot use regex constraint with JSON response format",
        )?;

        utils::validate_conflicting_parameters(
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

        utils::validate_mutually_exclusive_options(
            &structured_constraints,
            "Only one structured output constraint (regex, ebnf, or json_schema) can be active at a time",
        )?;

        Ok(())
    }
}

impl ValidatableRequest for ChatCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        // Call the common validation function from the validation module
        utils::validate_common_request_params(self)?;

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
    use super::constants::*;
    use super::utils::*;
    use super::*;
    use crate::protocols::spec::StringOrArray;

    // Mock request type for testing validation traits
    #[derive(Debug, Default)]
    struct MockRequest {
        temperature: Option<f32>,
        stop: Option<StringOrArray>,
        max_tokens: Option<u32>,
        min_tokens: Option<u32>,
    }

    impl SamplingOptionsProvider for MockRequest {
        fn get_temperature(&self) -> Option<f32> {
            self.temperature
        }
        fn get_top_p(&self) -> Option<f32> {
            None
        }
        fn get_frequency_penalty(&self) -> Option<f32> {
            None
        }
        fn get_presence_penalty(&self) -> Option<f32> {
            None
        }
    }

    impl StopConditionsProvider for MockRequest {
        fn get_stop_sequences(&self) -> Option<&StringOrArray> {
            self.stop.as_ref()
        }
    }

    impl TokenLimitsProvider for MockRequest {
        fn get_max_tokens(&self) -> Option<u32> {
            self.max_tokens
        }
        fn get_min_tokens(&self) -> Option<u32> {
            self.min_tokens
        }
    }

    impl LogProbsProvider for MockRequest {
        fn get_logprobs(&self) -> Option<u32> {
            None
        }
        fn get_top_logprobs(&self) -> Option<u32> {
            None
        }
    }

    impl SGLangExtensionsProvider for MockRequest {}
    impl CompletionCountProvider for MockRequest {}
    impl ValidatableRequest for MockRequest {}

    #[test]
    fn test_range_validation() {
        // Valid range
        assert!(validate_range(1.5f32, &TEMPERATURE_RANGE, "temperature").is_ok());
        // Invalid range
        assert!(validate_range(-0.1f32, &TEMPERATURE_RANGE, "temperature").is_err());
        assert!(validate_range(3.0f32, &TEMPERATURE_RANGE, "temperature").is_err());
    }

    #[test]
    fn test_sglang_top_k_validation() {
        assert!(validate_top_k(-1).is_ok()); // Disabled
        assert!(validate_top_k(50).is_ok()); // Valid positive
        assert!(validate_top_k(0).is_err()); // Invalid
        assert!(validate_top_k(-5).is_err()); // Invalid
    }

    #[test]
    fn test_stop_sequences_limits() {
        let request = MockRequest {
            stop: Some(StringOrArray::Array(vec![
                "stop1".to_string(),
                "stop2".to_string(),
                "stop3".to_string(),
                "stop4".to_string(),
                "stop5".to_string(), // Too many
            ])),
            ..Default::default()
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn test_token_limits_conflict() {
        let request = MockRequest {
            min_tokens: Some(100),
            max_tokens: Some(50), // min > max
            ..Default::default()
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn test_valid_request() {
        let request = MockRequest {
            temperature: Some(1.0),
            stop: Some(StringOrArray::Array(vec!["stop".to_string()])),
            max_tokens: Some(100),
            min_tokens: Some(10),
        };
        assert!(request.validate().is_ok());
    }

    // Chat completion specific tests
    #[cfg(test)]
    mod chat_tests {
        use super::*;

        fn create_valid_chat_request() -> ChatCompletionRequest {
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
                chat_template_kwargs: None,
                return_hidden_states: false,
            }
        }

        #[test]
        fn test_chat_validation_basics() {
            // Valid request
            assert!(create_valid_chat_request().validate().is_ok());

            // Empty messages
            let mut request = create_valid_chat_request();
            request.messages = vec![];
            assert!(request.validate().is_err());

            // Invalid temperature
            let mut request = create_valid_chat_request();
            request.temperature = Some(3.0);
            assert!(request.validate().is_err());
        }

        #[test]
        fn test_chat_conflicts() {
            let mut request = create_valid_chat_request();

            // Conflicting max_tokens
            request.max_tokens = Some(100);
            request.max_completion_tokens = Some(200);
            assert!(request.validate().is_err());

            // Logprobs without top_logprobs
            request.max_tokens = None;
            request.logprobs = true;
            request.top_logprobs = None;
            assert!(request.validate().is_err());
        }

        #[test]
        fn test_sglang_extensions() {
            let mut request = create_valid_chat_request();

            // Valid SGLang parameters
            request.top_k = Some(-1);
            request.min_p = Some(0.1);
            request.repetition_penalty = Some(1.2);
            assert!(request.validate().is_ok());

            // Invalid parameters
            request.top_k = Some(0); // Invalid
            assert!(request.validate().is_err());
        }

        #[test]
        fn test_parameter_ranges() {
            let mut request = create_valid_chat_request();

            // Test temperature range (0.0 to 2.0)
            request.temperature = Some(1.5);
            assert!(request.validate().is_ok());
            request.temperature = Some(-0.1);
            assert!(request.validate().is_err());
            request.temperature = Some(3.0);
            assert!(request.validate().is_err());

            // Test top_p range (0.0 to 1.0)
            request.temperature = Some(1.0); // Reset
            request.top_p = Some(0.9);
            assert!(request.validate().is_ok());
            request.top_p = Some(-0.1);
            assert!(request.validate().is_err());
            request.top_p = Some(1.5);
            assert!(request.validate().is_err());

            // Test frequency_penalty range (-2.0 to 2.0)
            request.top_p = Some(0.9); // Reset
            request.frequency_penalty = Some(1.5);
            assert!(request.validate().is_ok());
            request.frequency_penalty = Some(-2.5);
            assert!(request.validate().is_err());
            request.frequency_penalty = Some(3.0);
            assert!(request.validate().is_err());

            // Test presence_penalty range (-2.0 to 2.0)
            request.frequency_penalty = Some(0.0); // Reset
            request.presence_penalty = Some(-1.5);
            assert!(request.validate().is_ok());
            request.presence_penalty = Some(-3.0);
            assert!(request.validate().is_err());
            request.presence_penalty = Some(2.5);
            assert!(request.validate().is_err());

            // Test repetition_penalty range (0.0 to 2.0)
            request.presence_penalty = Some(0.0); // Reset
            request.repetition_penalty = Some(1.2);
            assert!(request.validate().is_ok());
            request.repetition_penalty = Some(-0.1);
            assert!(request.validate().is_err());
            request.repetition_penalty = Some(2.1);
            assert!(request.validate().is_err());

            // Test min_p range (0.0 to 1.0)
            request.repetition_penalty = Some(1.0); // Reset
            request.min_p = Some(0.5);
            assert!(request.validate().is_ok());
            request.min_p = Some(-0.1);
            assert!(request.validate().is_err());
            request.min_p = Some(1.5);
            assert!(request.validate().is_err());
        }

        #[test]
        fn test_structured_output_conflicts() {
            let mut request = create_valid_chat_request();

            // JSON response format with regex should conflict
            request.response_format = Some(ResponseFormat::JsonObject);
            request.regex = Some(".*".to_string());
            assert!(request.validate().is_err());

            // JSON response format with EBNF should conflict
            request.regex = None;
            request.ebnf = Some("grammar".to_string());
            assert!(request.validate().is_err());

            // Multiple structured constraints should conflict
            request.response_format = None;
            request.regex = Some(".*".to_string());
            request.ebnf = Some("grammar".to_string());
            assert!(request.validate().is_err());

            // Only one constraint should work
            request.ebnf = None;
            request.regex = Some(".*".to_string());
            assert!(request.validate().is_ok());

            request.regex = None;
            request.ebnf = Some("grammar".to_string());
            assert!(request.validate().is_ok());

            request.ebnf = None;
            request.response_format = Some(ResponseFormat::JsonObject);
            assert!(request.validate().is_ok());
        }

        #[test]
        fn test_stop_sequences_validation() {
            let mut request = create_valid_chat_request();

            // Valid stop sequences
            request.stop = Some(StringOrArray::Array(vec![
                "stop1".to_string(),
                "stop2".to_string(),
            ]));
            assert!(request.validate().is_ok());

            // Too many stop sequences (max 4)
            request.stop = Some(StringOrArray::Array(vec![
                "stop1".to_string(),
                "stop2".to_string(),
                "stop3".to_string(),
                "stop4".to_string(),
                "stop5".to_string(),
            ]));
            assert!(request.validate().is_err());

            // Empty stop sequence should fail
            request.stop = Some(StringOrArray::String("".to_string()));
            assert!(request.validate().is_err());

            // Empty string in array should fail
            request.stop = Some(StringOrArray::Array(vec![
                "stop1".to_string(),
                "".to_string(),
            ]));
            assert!(request.validate().is_err());
        }

        #[test]
        fn test_logprobs_validation() {
            let mut request = create_valid_chat_request();

            // Valid logprobs configuration
            request.logprobs = true;
            request.top_logprobs = Some(10);
            assert!(request.validate().is_ok());

            // logprobs=true without top_logprobs should fail
            request.top_logprobs = None;
            assert!(request.validate().is_err());

            // top_logprobs without logprobs=true should fail
            request.logprobs = false;
            request.top_logprobs = Some(10);
            assert!(request.validate().is_err());

            // top_logprobs out of range (0-20)
            request.logprobs = true;
            request.top_logprobs = Some(25);
            assert!(request.validate().is_err());
        }

        #[test]
        fn test_n_parameter_validation() {
            let mut request = create_valid_chat_request();

            // Valid n values (1-10)
            request.n = Some(1);
            assert!(request.validate().is_ok());
            request.n = Some(5);
            assert!(request.validate().is_ok());
            request.n = Some(10);
            assert!(request.validate().is_ok());

            // Invalid n values
            request.n = Some(0);
            assert!(request.validate().is_err());
            request.n = Some(15);
            assert!(request.validate().is_err());
        }

        #[test]
        fn test_min_max_tokens_validation() {
            let mut request = create_valid_chat_request();

            // Valid token limits
            request.min_tokens = Some(10);
            request.max_tokens = Some(100);
            assert!(request.validate().is_ok());

            // min_tokens > max_tokens should fail
            request.min_tokens = Some(150);
            request.max_tokens = Some(100);
            assert!(request.validate().is_err());

            // Should work with max_completion_tokens instead
            request.max_tokens = None;
            request.max_completion_tokens = Some(200);
            request.min_tokens = Some(50);
            assert!(request.validate().is_ok());

            // min_tokens > max_completion_tokens should fail
            request.min_tokens = Some(250);
            assert!(request.validate().is_err());
        }
    }
}
