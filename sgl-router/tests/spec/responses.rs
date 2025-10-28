use sglang_router_rs::protocols::responses::{ResponseInput, ResponsesRequest};
use validator::Validate;

/// Test that valid conversation IDs pass validation
#[test]
fn test_validate_conversation_id_valid() {
    let valid_ids = vec![
        "conv_123",
        "conv_test-123_abc",
        "conv_ABC_123",
        "conv_my_conversation_123",
        "conv_456",
        "conv_test123",
    ];

    for id in valid_ids {
        let request = ResponsesRequest {
            conversation: Some(id.to_string()),
            input: ResponseInput::Text("test".to_string()),
            ..Default::default()
        };
        assert!(
            request.validate().is_ok(),
            "Expected '{}' to be valid, but got error: {:?}",
            id,
            request.validate().err()
        );
    }
}

/// Test that invalid conversation IDs fail validation
#[test]
fn test_validate_conversation_id_invalid() {
    let invalid_ids = vec![
        // Missing 'conv_' prefix
        "test-conv-streaming",
        "conversation-456",
        "my_conversation_123",
        "ABC123",
        "test_123_conv",
        "conv123", // missing underscore
        // Invalid characters
        "conv_.test",     // contains dot
        "conv_ test",     // contains space
        "conv_@test",     // contains @
        "conv_/test",     // contains /
        "conv_\\test",    // contains backslash
        "conv_:test",     // contains colon
        "conv_;test",     // contains semicolon
        "conv_,test",     // contains comma
        "conv_+test",     // contains plus
        "conv_=test",     // contains equals
        "conv_[test]",    // contains brackets
        "conv_{test}",    // contains braces
        "conv_(test)",    // contains parentheses
        "conv_!test",     // contains exclamation
        "conv_?test",     // contains question mark
        "conv_#test",     // contains hash
        "conv_$test",     // contains dollar sign
        "conv_%test",     // contains percent
        "conv_&test",     // contains ampersand
        "conv_*test",     // contains asterisk
        "conv_ test-123", // contains space
    ];

    for id in invalid_ids {
        let request = ResponsesRequest {
            conversation: Some(id.to_string()),
            input: ResponseInput::Text("test".to_string()),
            ..Default::default()
        };
        let result = request.validate();
        assert!(
            result.is_err(),
            "Expected '{}' to be invalid, but validation passed",
            id
        );

        // Verify error is for conversation field
        if let Err(errors) = result {
            let field_errors = errors.field_errors();
            let conversation_errors = field_errors.get("conversation");
            assert!(
                conversation_errors.is_some(),
                "Expected error for 'conversation' field, but got errors for: {:?}",
                field_errors.keys()
            );

            let error_msg = conversation_errors
                .and_then(|errs| errs.first())
                .and_then(|err| err.message.as_ref())
                .map(|msg| msg.to_string());

            assert!(
                error_msg.is_some(),
                "Expected error message for conversation field"
            );
            let msg = error_msg.unwrap();
            assert!(
                msg.contains("Invalid 'conversation'"),
                "Error message should mention 'conversation', got: {}",
                msg
            );
            assert!(
                msg.contains(id),
                "Error message should include the invalid ID '{}', got: {}",
                id,
                msg
            );
        }
    }
}

/// Test that None conversation ID is valid
#[test]
fn test_validate_conversation_id_none() {
    let request = ResponsesRequest {
        conversation: None,
        input: ResponseInput::Text("test".to_string()),
        ..Default::default()
    };
    assert!(
        request.validate().is_ok(),
        "Request with no conversation ID should be valid"
    );
}

/// Test the exact error format matches OpenAI's error message for invalid characters
#[test]
fn test_validate_conversation_id_error_message_format() {
    let invalid_id = "conv_.test-conv-streaming";
    let request = ResponsesRequest {
        conversation: Some(invalid_id.to_string()),
        input: ResponseInput::Text("test".to_string()),
        ..Default::default()
    };

    let result = request.validate();
    assert!(result.is_err());

    if let Err(errors) = result {
        let error_msg = errors
            .field_errors()
            .get("conversation")
            .and_then(|errs| errs.first())
            .and_then(|err| err.message.as_ref())
            .map(|msg| msg.to_string())
            .unwrap();

        // Verify the error message matches OpenAI's format
        assert!(
            error_msg.starts_with("Invalid 'conversation':"),
            "Error should start with \"Invalid 'conversation':\""
        );
        assert!(
            error_msg.contains("letters, numbers, underscores, or dashes"),
            "Error should mention valid characters"
        );
        assert!(
            error_msg.contains(invalid_id),
            "Error should include the invalid conversation ID"
        );
    }
}

/// Test the exact error format for missing 'conv_' prefix
#[test]
fn test_validate_conversation_id_missing_prefix() {
    let invalid_id = "test-conv-streaming";
    let request = ResponsesRequest {
        conversation: Some(invalid_id.to_string()),
        input: ResponseInput::Text("test".to_string()),
        ..Default::default()
    };

    let result = request.validate();
    assert!(result.is_err());

    if let Err(errors) = result {
        let error_msg = errors
            .field_errors()
            .get("conversation")
            .and_then(|errs| errs.first())
            .and_then(|err| err.message.as_ref())
            .map(|msg| msg.to_string())
            .unwrap();

        // Verify the error message matches OpenAI's format
        assert!(
            error_msg.starts_with("Invalid 'conversation':"),
            "Error should start with \"Invalid 'conversation':\""
        );
        assert!(
            error_msg.contains("begins with 'conv_'"),
            "Error should mention the required prefix, got: {}",
            error_msg
        );
        assert!(
            error_msg.contains(invalid_id),
            "Error should include the invalid conversation ID"
        );
    }
}
