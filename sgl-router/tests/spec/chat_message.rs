use serde_json::json;
use sglang_router_rs::protocols::{
    chat::{ChatMessage, UserMessageContent},
    common::ContentPart,
};

#[test]
fn test_chat_message_tagged_by_role_system() {
    let json = json!({
        "role": "system",
        "content": "You are a helpful assistant"
    });

    let msg: ChatMessage = serde_json::from_value(json).unwrap();
    match msg {
        ChatMessage::System { content, .. } => match content {
            UserMessageContent::Text(text) => assert_eq!(text, "You are a helpful assistant"),
            _ => panic!("Expected Text variant"),
        },
        _ => panic!("Expected System variant"),
    }
}

#[test]
fn test_chat_message_tagged_by_role_system_content_is_list() {
    let json = json!({
        "role": "system",
        "content":
        [{
            "type": "text",
            "text": "You are a helpful assistant"
        }]
    });

    let msg: ChatMessage = serde_json::from_value(json).unwrap();
    match msg {
        ChatMessage::System { content, .. } => {
            let text = content.extract_text_from_content();
            assert_eq!(text, "You are a helpful assistant");
        }
        _ => panic!("Expected System variant"),
    }
}

#[test]
fn test_chat_message_tagged_by_role_user_content_is_list() {
    let json = json!({
        "role": "user",
        "content":
        [{
            "type": "text",
            "text": "You are a helpful assistant"
        }]
    });

    let msg: ChatMessage = serde_json::from_value(json).unwrap();
    match msg {
        ChatMessage::User { content, .. } => {
            match &content {
                UserMessageContent::Parts(parts) => {
                    assert_eq!(parts.len(), 1);
                    match &parts[0] {
                        ContentPart::Text { text } => {
                            assert_eq!(text, "You are a helpful assistant");
                        }
                        _ => panic!("Expected Text variant"),
                    }
                }
                _ => panic!("Expected Parts variant"),
            }
            let text = content.extract_text_from_content();
            assert_eq!(text, "You are a helpful assistant");
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
            UserMessageContent::Text(text) => assert_eq!(text, "Hello"),
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
            assert_eq!(content, Some("Hi there!".to_string()));
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
            assert_eq!(content, "Tool result");
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
