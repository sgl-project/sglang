//! Built-in tool detection and validation
//!
//! Scans request tools to identify built-in tools and validates that they
//! don't have unsupported configuration (e.g., custom server_url, authorization).

use super::types::BuiltinToolType;
use crate::protocols::responses::{ResponseTool, ResponseToolType};

/// Detects built-in tools in a request
pub struct BuiltinToolDetector;

impl BuiltinToolDetector {
    /// Scan tools and extract built-in tool types
    ///
    /// Returns Err if built-in tools have invalid configuration
    /// (e.g., server_url or authorization specified)
    pub fn detect(tools: &[ResponseTool]) -> Result<Vec<BuiltinToolType>, String> {
        let mut builtin_tools = Vec::new();

        for tool in tools {
            if let Some(tool_type) = BuiltinToolType::from_response_tool(tool) {
                // VALIDATION: Built-in tools must NOT have server_url or authorization
                if tool.server_url.is_some() || tool.authorization.is_some() {
                    return Err(format!(
                        "Built-in tool '{}' does not support 'server_url' or 'authorization' fields",
                        tool_type.fixed_label()
                    ));
                }

                // Note: 'label' field doesn't exist in ResponseTool
                // Labels are fixed per tool type internally

                builtin_tools.push(tool_type);
            }
        }

        Ok(builtin_tools)
    }

    /// Check if any built-in tools are present
    ///
    /// This is a fast check to determine if built-in tool processing is needed.
    pub fn has_builtin_tools(tools: &[ResponseTool]) -> bool {
        tools.iter().any(|tool| {
            matches!(
                tool.r#type,
                ResponseToolType::WebSearch
                    | ResponseToolType::FileSearch
                    | ResponseToolType::CodeInterpreter
            )
        })
    }

    /// Validate that built-in tools don't have unsupported fields
    pub fn validate_no_custom_config(tools: &[ResponseTool]) -> Result<(), String> {
        for tool in tools {
            if let Some(tool_type) = BuiltinToolType::from_response_tool(tool) {
                if tool.server_url.is_some() || tool.authorization.is_some() {
                    return Err(format!(
                        "Built-in tool '{}' does not support 'server_url' or 'authorization' fields",
                        tool_type.fixed_label()
                    ));
                }
            }
        }
        Ok(())
    }
}
