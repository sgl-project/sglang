use serde_json::json;
use sglang_router_rs::protocols::spec::ChatCompletionRequest;
use sglang_router_rs::protocols::spec::ChatMessage;
#[tokio::test]
async fn test_parse_request() {
    let request_body = json!({
      "messages": [
        {
          "content": [
            {
              "type": "text",
              "text": "..."
            }
          ],
          "role": "system"
        },
        {
          "content": [
            {
              "type": "text",
              "text": "..."
            }
          ],
          "role": "user"
        },
        {
          "content": [
            {
              "type": "text",
              "text": "..."
            }
          ],
          "role": "assistant",
          "tool_calls": [
            {
              "id": "xxx",
              "type": "function",
              "function": {
                "name": "think",
                "arguments": "{
                    \"thought\": \"...\"
                  }"
              }
            }
          ],
          "some_future_keys": "some_future_values",
        },
        {
          "content": [
            {
              "type": "text",
              "text": "..."
            }
          ],
          "role": "tool",
          "tool_call_id": "xxx",
        }
      ],
      "model": "xxx",
      "max_completion_tokens": 4096,
      "temperature": 0.0,
      "tools": [],
    });

    let request_body_str = serde_json::to_string(&request_body).unwrap();

    let parsed_request: ChatCompletionRequest = serde_json::from_str(&request_body_str).unwrap();

    assert_eq!(parsed_request.messages.len(), 4);

    assert!(matches!(
        &parsed_request.messages[0],
        ChatMessage::System { .. }
    ));
    assert!(matches!(
        &parsed_request.messages[1],
        ChatMessage::User { .. }
    ));
    assert!(matches!(
        &parsed_request.messages[2],
        ChatMessage::Assistant { .. }
    ));
    assert!(matches!(
        &parsed_request.messages[3],
        ChatMessage::Tool { .. }
    ));

    let serialized_request_str = serde_json::to_string(&parsed_request).unwrap();
    let original_value: serde_json::Value = serde_json::from_str(&request_body_str).unwrap();
    let serialized_value: serde_json::Value =
        serde_json::from_str(&serialized_request_str).unwrap();

    compare_json_values(&original_value, &serialized_value);
}

fn compare_json_values(original: &serde_json::Value, serialized: &serde_json::Value) {
    match original {
        serde_json::Value::Object(original_obj) => {
            if let serde_json::Value::Object(serialized_obj) = serialized {
                for (key, original_value) in original_obj {
                    assert!(
                        serialized_obj.contains_key(key),
                        "Serialized value should contain key '{}'",
                        key
                    );
                    let serialized_value = &serialized_obj[key];
                    compare_json_values(original_value, serialized_value);
                }
            } else {
                panic!("Expected serialized value to be an object");
            }
        }
        serde_json::Value::Array(original_array) => {
            if let serde_json::Value::Array(serialized_array) = serialized {
                assert_eq!(
                    original_array.len(),
                    serialized_array.len(),
                    "Array lengths should match"
                );
                for (original_item, serialized_item) in
                    original_array.iter().zip(serialized_array.iter())
                {
                    compare_json_values(original_item, serialized_item);
                }
            } else {
                panic!("Expected serialized value to be an array");
            }
        }
        _ => {
            assert_eq!(original, serialized, "Values should be equal");
        }
    }
}
