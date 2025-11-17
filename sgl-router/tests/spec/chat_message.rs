use serde_json::json;
use sglang_router_rs::protocols::chat::{ChatMessage, MessageContent};

#[test]
fn test_chat_message_tagged_by_role_system() {
    let json = json!({
        "role": "system",
        "content": "You are a helpful assistant"
    });

    let msg: ChatMessage = serde_json::from_value(json).unwrap();
    match msg {
        ChatMessage::System { content, .. } => {
            assert_eq!(
                content,
                MessageContent::Text("You are a helpful assistant".to_string())
            )
        }
        _ => panic!("Expected System variant"),
    }
}

#[test]
fn test_chat_message_tagged_by_role_user() {
    let json = json!({
        "role": "user",
        "content": "Hello"
    });

    let msg: ChatMessage = serde_json::from_value(json).unwrap();
    match msg {
        ChatMessage::User { content, .. } => match content {
            MessageContent::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text content"),
        },
        _ => panic!("Expected User variant"),
    }
}

#[test]
fn test_chat_message_tagged_by_role_assistant() {
    let json = json!({
        "role": "assistant",
        "content": "Hi there!"
    });

    let msg: ChatMessage = serde_json::from_value(json).unwrap();
    match msg {
        ChatMessage::Assistant { content, .. } => {
            assert_eq!(content, Some(MessageContent::Text("Hi there!".to_string())));
        }
        _ => panic!("Expected Assistant variant"),
    }
}

#[test]
fn test_chat_message_tagged_by_role_tool() {
    let json = json!({
        "role": "tool",
        "content": "Tool result",
        "tool_call_id": "call_123"
    });

    let msg: ChatMessage = serde_json::from_value(json).unwrap();
    match msg {
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => {
            match content {
                MessageContent::Text(text) => {
                    assert_eq!(text, "Tool result");
                }
                _ => panic!("Expected content to be a string"),
            }
            assert_eq!(tool_call_id, "call_123");
        }
        _ => panic!("Expected Tool variant"),
    }
}

#[test]
fn test_chat_message_wrong_role_rejected() {
    let json = json!({
        "role": "invalid_role",
        "content": "test"
    });

    let result = serde_json::from_value::<ChatMessage>(json);
    assert!(result.is_err(), "Should reject invalid role");
}
