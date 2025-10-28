use sglang_router_rs::protocols::responses::{ResponseInput, ResponsesRequest};
use validator::Validate;

/// Test that valid conversation IDs pass validation
#[test]
fn test_validate_conversation_id_valid() {
    let valid_ids = vec![
        "conv_123",
        "conversation-456",
        "my_conversation_123",
        "conv-test-123_abc",
        "ABC123",
        "test_123_conv",
        "conv123",
        "CONV_ABC_123",
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
        "conv.test",     // contains dot
        "conv test",     // contains space
        "conv@test",     // contains @
        "conv/test",     // contains /
        "conv\\test",    // contains backslash
        "conv:test",     // contains colon
        "conv;test",     // contains semicolon
        "conv,test",     // contains comma
        "conv+test",     // contains plus
        "conv=test",     // contains equals
        "conv[test]",    // contains brackets
        "conv{test}",    // contains braces
        "conv(test)",    // contains parentheses
        "conv!test",     // contains exclamation
        "conv?test",     // contains question mark
        "conv#test",     // contains hash
        "conv$test",     // contains dollar sign
        "conv%test",     // contains percent
        "conv&test",     // contains ampersand
        "conv*test",     // contains asterisk
        "conv test-123", // contains space
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

/// Test the exact error format matches OpenAI's error message
#[test]
fn test_validate_conversation_id_error_message_format() {
    let invalid_id = "conv.test-conv-streaming";
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
