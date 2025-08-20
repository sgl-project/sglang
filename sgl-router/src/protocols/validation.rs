// Core validation infrastructure for API parameter validation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

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
                crate::protocols::common::StringOrArray::String(s) => {
                    if s.is_empty() {
                        return Err(ValidationError::InvalidValue {
                            parameter: "stop".to_string(),
                            value: "empty string".to_string(),
                            reason: "stop sequences cannot be empty".to_string(),
                        });
                    }
                }
                crate::protocols::common::StringOrArray::Array(arr) => {
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
    fn get_stop_sequences(&self) -> Option<&crate::protocols::common::StringOrArray>;
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

#[cfg(test)]
mod tests {
    use super::constants::*;
    use super::utils::*;
    use super::*;
    use crate::protocols::common::StringOrArray;

    // Mock request type for testing validation traits
    #[derive(Debug, Default)]
    struct MockRequest {
        temperature: Option<f32>,
        top_p: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Option<StringOrArray>,
        max_tokens: Option<u32>,
        min_tokens: Option<u32>,
        logprobs: Option<u32>,
        top_logprobs: Option<u32>,
    }

    impl SamplingOptionsProvider for MockRequest {
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
            self.logprobs
        }
        fn get_top_logprobs(&self) -> Option<u32> {
            self.top_logprobs
        }
    }

    impl SGLangExtensionsProvider for MockRequest {
        // Default implementations return None, so no custom logic needed
    }

    impl CompletionCountProvider for MockRequest {
        // Default implementation returns None, so no custom logic needed
    }

    impl ValidatableRequest for MockRequest {}

    #[test]
    fn test_validate_range_valid() {
        let result = validate_range(1.5f32, &TEMPERATURE_RANGE, "temperature");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.5f32);
    }

    #[test]
    fn test_validate_range_too_low() {
        let result = validate_range(-0.1f32, &TEMPERATURE_RANGE, "temperature");
        assert!(result.is_err());
        match result.unwrap_err() {
            ValidationError::OutOfRange { parameter, .. } => {
                assert_eq!(parameter, "temperature");
            }
            _ => panic!("Expected OutOfRange error"),
        }
    }

    #[test]
    fn test_validate_positive_valid() {
        let result = validate_positive(5i32, "max_tokens");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5i32);
    }

    #[test]
    fn test_validate_max_items_valid() {
        let items = vec!["stop1", "stop2"];
        let result = validate_max_items(&items, MAX_STOP_SEQUENCES, "stop");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_top_k() {
        assert!(validate_top_k(-1).is_ok()); // Disabled
        assert!(validate_top_k(50).is_ok()); // Positive
        assert!(validate_top_k(0).is_err()); // Invalid
        assert!(validate_top_k(-5).is_err()); // Invalid
    }

    #[test]
    fn test_valid_request() {
        let request = MockRequest {
            temperature: Some(1.0),
            top_p: Some(0.9),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(-0.5),
            stop: Some(StringOrArray::Array(vec![
                "stop1".to_string(),
                "stop2".to_string(),
            ])),
            max_tokens: Some(100),
            min_tokens: Some(10),
            logprobs: Some(3),
            top_logprobs: Some(15),
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_invalid_temperature() {
        let request = MockRequest {
            temperature: Some(3.0), // Invalid: too high
            ..Default::default()
        };

        let result = request.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_too_many_stop_sequences() {
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

        let result = request.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            ValidationError::TooManyItems {
                parameter,
                count,
                max,
            } => {
                assert_eq!(parameter, "stop");
                assert_eq!(count, 5);
                assert_eq!(max, MAX_STOP_SEQUENCES);
            }
            _ => panic!("Expected TooManyItems error"),
        }
    }

    #[test]
    fn test_conflicting_token_limits() {
        let request = MockRequest {
            min_tokens: Some(100),
            max_tokens: Some(50), // Invalid: min > max
            ..Default::default()
        };

        let result = request.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            ValidationError::ConflictingParameters {
                parameter1,
                parameter2,
                ..
            } => {
                assert_eq!(parameter1, "min_tokens");
                assert_eq!(parameter2, "max_tokens");
            }
            _ => panic!("Expected ConflictingParameters error"),
        }
    }

    #[test]
    fn test_boundary_values() {
        let request = MockRequest {
            temperature: Some(0.0),        // Boundary: minimum
            top_p: Some(1.0),              // Boundary: maximum
            frequency_penalty: Some(-2.0), // Boundary: minimum
            presence_penalty: Some(2.0),   // Boundary: maximum
            logprobs: Some(0),             // Boundary: minimum
            top_logprobs: Some(20),        // Boundary: maximum
            ..Default::default()
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_validation_error_display() {
        let error = ValidationError::OutOfRange {
            parameter: "temperature".to_string(),
            value: "3.0".to_string(),
            min: "0.0".to_string(),
            max: "2.0".to_string(),
        };

        let message = format!("{}", error);
        assert!(message.contains("temperature"));
        assert!(message.contains("3.0"));
    }
}
