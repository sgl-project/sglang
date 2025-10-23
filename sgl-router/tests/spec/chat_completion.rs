use serde_json::json;
use sglang_router_rs::protocols::{
    chat::{ChatCompletionRequest, ChatMessage, UserMessageContent},
    common::{
        Function, FunctionCall, FunctionChoice, StreamOptions, Tool, ToolChoice, ToolChoiceValue,
        ToolReference,
    },
    validated::Normalizable,
};
use validator::Validate;

// Deprecated fields normalization tests

#[test]
fn test_max_tokens_normalizes_to_max_completion_tokens() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        max_tokens: Some(100),
        max_completion_tokens: None,
        ..Default::default()
    };

    req.normalize();
    assert_eq!(
        req.max_completion_tokens,
        Some(100),
        "max_tokens should be copied to max_completion_tokens"
    );
    #[allow(deprecated)]
    {
        assert!(
            req.max_tokens.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_max_completion_tokens_takes_precedence() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        max_tokens: Some(100),
        max_completion_tokens: Some(200),
        ..Default::default()
    };

    req.normalize();
    assert_eq!(
        req.max_completion_tokens,
        Some(200),
        "max_completion_tokens should take precedence"
    );
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_functions_normalizes_to_tools() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        functions: Some(vec![Function {
            name: "test_func".to_string(),
            description: Some("Test function".to_string()),
            parameters: json!({}),
            strict: None,
        }]),
        tools: None,
        ..Default::default()
    };

    req.normalize();
    assert!(req.tools.is_some(), "functions should be migrated to tools");
    assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    assert_eq!(req.tools.as_ref().unwrap()[0].function.name, "test_func");
    #[allow(deprecated)]
    {
        assert!(
            req.functions.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_function_call_normalizes_to_tool_choice() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        function_call: Some(FunctionCall::None),
        tool_choice: None,
        ..Default::default()
    };

    req.normalize();
    assert!(
        req.tool_choice.is_some(),
        "function_call should be migrated to tool_choice"
    );
    assert!(matches!(
        req.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    ));
    #[allow(deprecated)]
    {
        assert!(
            req.function_call.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

#[test]
fn test_function_call_function_variant_normalizes() {
    #[allow(deprecated)]
    let mut req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        function_call: Some(FunctionCall::Function {
            name: "my_function".to_string(),
        }),
        tool_choice: None,
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "my_function".to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            },
        }]),
        ..Default::default()
    };

    req.normalize();
    assert!(
        req.tool_choice.is_some(),
        "function_call should be migrated to tool_choice"
    );
    match &req.tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            assert_eq!(function.name, "my_function");
        }
        _ => panic!("Expected ToolChoice::Function variant"),
    }
    #[allow(deprecated)]
    {
        assert!(
            req.function_call.is_none(),
            "Deprecated field should be cleared"
        );
    }
    assert!(
        req.validate().is_ok(),
        "Should be valid after normalization"
    );
}

// Stream options validation tests

#[test]
fn test_stream_options_requires_stream_enabled() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        stream: false,
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_err(),
        "Should reject stream_options when stream is false"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("stream_options") && err.contains("stream") && err.contains("enabled"),
        "Error should mention stream dependency: {}",
        err
    );
}

#[test]
fn test_stream_options_valid_when_stream_enabled() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        stream: true,
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_ok(),
        "Should accept stream_options when stream is true"
    );
}

#[test]
fn test_no_stream_options_valid_when_stream_disabled() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        stream: false,
        stream_options: None,
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_ok(),
        "Should accept no stream_options when stream is false"
    );
}

// Tool choice validation tests
#[test]
fn test_tool_choice_function_not_found() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::Function {
            function: FunctionChoice {
                name: "nonexistent_function".to_string(),
            },
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_err(), "Should reject nonexistent function name");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("function 'nonexistent_function' not found"),
        "Error should mention the missing function: {}",
        err
    );
}

#[test]
fn test_tool_choice_function_exists_valid() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::Function {
            function: FunctionChoice {
                name: "get_weather".to_string(),
            },
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept existing function name");
}

#[test]
fn test_tool_choice_allowed_tools_invalid_mode() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "invalid_mode".to_string(),
            tools: vec![ToolReference {
                tool_type: "function".to_string(),
                name: "get_weather".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_err(), "Should reject invalid mode");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("must be 'auto' or 'required'"),
        "Error should mention valid modes: {}",
        err
    );
}

#[test]
fn test_tool_choice_allowed_tools_valid_mode_auto() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![ToolReference {
                tool_type: "function".to_string(),
                name: "get_weather".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept 'auto' mode");
}

#[test]
fn test_tool_choice_allowed_tools_valid_mode_required() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "required".to_string(),
            tools: vec![ToolReference {
                tool_type: "function".to_string(),
                name: "get_weather".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept 'required' mode");
}

#[test]
fn test_tool_choice_allowed_tools_tool_not_found() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({}),
                strict: None,
            },
        }]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![ToolReference {
                tool_type: "function".to_string(),
                name: "nonexistent_tool".to_string(),
            }],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_err(), "Should reject nonexistent tool name");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("tool 'nonexistent_tool' not found"),
        "Error should mention the missing tool: {}",
        err
    );
}

#[test]
fn test_tool_choice_allowed_tools_multiple_tools_valid() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_time".to_string(),
                    description: Some("Get time".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
        ]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![
                ToolReference {
                    tool_type: "function".to_string(),
                    name: "get_weather".to_string(),
                },
                ToolReference {
                    tool_type: "function".to_string(),
                    name: "get_time".to_string(),
                },
            ],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(result.is_ok(), "Should accept all valid tool references");
}

#[test]
fn test_tool_choice_allowed_tools_one_invalid_among_valid() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("hello".to_string()),
            name: None,
        }],
        tools: Some(vec![
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_time".to_string(),
                    description: Some("Get time".to_string()),
                    parameters: json!({}),
                    strict: None,
                },
            },
        ]),
        tool_choice: Some(ToolChoice::AllowedTools {
            mode: "auto".to_string(),
            tools: vec![
                ToolReference {
                    tool_type: "function".to_string(),
                    name: "get_weather".to_string(),
                },
                ToolReference {
                    tool_type: "function".to_string(),
                    name: "nonexistent_tool".to_string(),
                },
            ],
            tool_type: "function".to_string(),
        }),
        ..Default::default()
    };

    let result = req.validate();
    assert!(
        result.is_err(),
        "Should reject if any tool reference is invalid"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("tool 'nonexistent_tool' not found"),
        "Error should mention the missing tool: {}",
        err
    );
}
