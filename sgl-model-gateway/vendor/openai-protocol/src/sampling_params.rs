use serde::{Deserialize, Serialize};
use validator::Validate;

use super::common::StringOrArray;

/// Sampling parameters for text generation
#[derive(Debug, Clone, Deserialize, Serialize, Default, Validate)]
#[validate(schema(function = "validate_sampling_params"))]
pub struct SamplingParams {
    /// Temperature for sampling (must be >= 0.0, no upper limit)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0))]
    pub temperature: Option<f32>,
    /// Maximum number of new tokens to generate (must be >= 0)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0))]
    pub max_new_tokens: Option<u32>,
    /// Top-p nucleus sampling (0.0 < top_p <= 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,
    /// Top-k sampling (-1 to disable, or >= 1)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,
    /// Minimum number of new tokens (validated in schema function for cross-field check with max_new_tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_stop_trim: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_seed: Option<u64>,
}

// ============================================================================
// Shared Validation Functions
// ============================================================================

/// Validates top_p: 0.0 < top_p <= 1.0 (can't use range validator for open interval)
pub fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

/// Validates top_k: -1 (disabled) or >= 1 (special -1 case - can't use range validator)
pub fn validate_top_k_value(top_k: i32) -> Result<(), validator::ValidationError> {
    if top_k != -1 && top_k < 1 {
        return Err(validator::ValidationError::new(
            "top_k must be -1 (disabled) or at least 1",
        ));
    }
    Ok(())
}

// ============================================================================
// SamplingParams-Specific Validation
// ============================================================================

/// Validation function for SamplingParams - cross-field validation only
fn validate_sampling_params(params: &SamplingParams) -> Result<(), validator::ValidationError> {
    // 1. Cross-field validation: min_new_tokens <= max_new_tokens
    if let (Some(min), Some(max)) = (params.min_new_tokens, params.max_new_tokens) {
        if min > max {
            return Err(validator::ValidationError::new(
                "min_new_tokens cannot exceed max_new_tokens",
            ));
        }
    }

    // 2. Validate mutually exclusive structured output constraints
    let constraint_count = [
        params.regex.is_some(),
        params.ebnf.is_some(),
        params.json_schema.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if constraint_count > 1 {
        return Err(validator::ValidationError::new(
            "only one of regex, ebnf, or json_schema can be set",
        ));
    }

    Ok(())
}
