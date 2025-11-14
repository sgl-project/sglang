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
    ///
    /// # Arguments
    /// * `tools` - List of tools from the request
    ///
    /// # Returns
    /// * `Ok(Vec<BuiltinToolType>)` - List of detected built-in tools
    /// * `Err(String)` - Validation error message
    ///
    /// # Example
    /// ```ignore
    /// let tools = vec![
    ///     ResponseTool { type: ResponseToolType::WebSearch, ..Default::default() }
    /// ];
    /// let builtin_types = BuiltinToolDetector::detect(&tools)?;
    /// assert_eq!(builtin_types, vec![BuiltinToolType::WebSearch]);
    /// ```
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
    ///
    /// # Example
    /// ```ignore
    /// if BuiltinToolDetector::has_builtin_tools(&tools) {
    ///     // Process built-in tools
    /// }
    /// ```
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
    ///
    /// This is a standalone validation method that can be called separately.
    ///
    /// # Returns
    /// * `Ok(())` - All built-in tools are valid
    /// * `Err(String)` - Validation error message
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_web_search() {
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearch,
            function: None,
            server_url: None,
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];

        let result = BuiltinToolDetector::detect(&tools).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], BuiltinToolType::WebSearch);
    }

    #[test]
    fn test_detect_multiple_builtin_tools() {
        let tools = vec![
            ResponseTool {
                r#type: ResponseToolType::WebSearch,
                function: None,
                server_url: None,
                authorization: None,
                server_label: None,
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            },
            ResponseTool {
                r#type: ResponseToolType::FileSearch,
                function: None,
                server_url: None,
                authorization: None,
                server_label: None,
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            },
        ];

        let result = BuiltinToolDetector::detect(&tools).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], BuiltinToolType::WebSearch);
        assert_eq!(result[1], BuiltinToolType::FileSearch);
    }

    #[test]
    fn test_reject_server_url() {
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearch,
            function: None,
            server_url: Some("https://custom-server.com".to_string()),
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];

        let result = BuiltinToolDetector::detect(&tools);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("does not support 'server_url'"));
    }

    #[test]
    fn test_reject_authorization() {
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearch,
            function: None,
            server_url: None,
            authorization: Some("Bearer token".to_string()),
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];

        let result = BuiltinToolDetector::detect(&tools);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("does not support 'server_url' or 'authorization'"));
    }

    #[test]
    fn test_has_builtin_tools_true() {
        let tools = vec![
            ResponseTool {
                r#type: ResponseToolType::Function,
                function: None,
                server_url: None,
                authorization: None,
                server_label: None,
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            },
            ResponseTool {
                r#type: ResponseToolType::WebSearch,
                function: None,
                server_url: None,
                authorization: None,
                server_label: None,
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            },
        ];

        assert!(BuiltinToolDetector::has_builtin_tools(&tools));
    }

    #[test]
    fn test_has_builtin_tools_false() {
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::Function,
            function: None,
            server_url: None,
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];

        assert!(!BuiltinToolDetector::has_builtin_tools(&tools));
    }

    #[test]
    fn test_validate_no_custom_config_valid() {
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearch,
            function: None,
            server_url: None,
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];

        assert!(BuiltinToolDetector::validate_no_custom_config(&tools).is_ok());
    }

    #[test]
    fn test_validate_no_custom_config_invalid() {
        let tools = vec![ResponseTool {
            r#type: ResponseToolType::WebSearch,
            function: None,
            server_url: Some("https://custom.com".to_string()),
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }];

        assert!(BuiltinToolDetector::validate_no_custom_config(&tools).is_err());
    }
}
