use sglang_router_rs::{
    protocols::{
        chat::{ChatMessage, MessageContent},
        common::{ContentPart, ImageUrl},
    },
    tokenizer::chat_template::{
        detect_chat_template_content_format, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateProcessor,
    },
};

#[test]
fn test_simple_chat_template() {
    let template = r#"
{%- for message in messages %}
<|{{ message.role }}|>{{ message.content }}<|end|>
{% endfor -%}
{%- if add_generation_prompt %}
<|assistant|>
{%- endif %}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = [ChatMessage::User {
        content: MessageContent::Text("Test".to_string()),
        name: None,
    }];

    // Convert to JSON values like the router does
    let message_values: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        ..Default::default()
    };
    let result = processor
        .apply_chat_template(&message_values, params)
        .unwrap();
    assert!(result.contains("<|user|>Test<|end|>"));
    assert!(result.contains("<|assistant|>"));
}

#[test]
fn test_chat_template_with_tokens() {
    // Template that uses template kwargs for tokens
    let template = r#"
{%- if bos_token -%}{{ bos_token }}{%- endif -%}
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}{%- if eos_token -%}{{ eos_token }}{%- endif -%}
{% endfor -%}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = [ChatMessage::User {
        content: MessageContent::Text("Test".to_string()),
        name: None,
    }];

    // Convert to JSON values like the router does
    let message_values: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    // Use template_kwargs to pass tokens
    let mut template_kwargs = std::collections::HashMap::new();
    template_kwargs.insert(
        "bos_token".to_string(),
        serde_json::Value::String("<s>".to_string()),
    );
    template_kwargs.insert(
        "eos_token".to_string(),
        serde_json::Value::String("</s>".to_string()),
    );

    let params = ChatTemplateParams {
        template_kwargs: Some(&template_kwargs),
        ..Default::default()
    };

    let result = processor
        .apply_chat_template(&message_values, params)
        .unwrap();
    assert!(result.contains("<s>"));
    assert!(result.contains("</s>"));
}

#[test]
fn test_llama_style_template() {
    let template = r#"
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
{%- endif -%}

{{- bos_token if bos_token else '<|begin_of_text|>' }}
{%- if system_message %}
{{- '<|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}
{%- endif %}

{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = [
        ChatMessage::System {
            content: MessageContent::Text("You are a helpful assistant".to_string()),
            name: None,
        },
        ChatMessage::User {
            content: MessageContent::Text("What is 2+2?".to_string()),
            name: None,
        },
    ];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    // Use template_kwargs to pass the token
    let mut template_kwargs = std::collections::HashMap::new();
    template_kwargs.insert(
        "bos_token".to_string(),
        serde_json::Value::String("<|begin_of_text|>".to_string()),
    );

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        template_kwargs: Some(&template_kwargs),
        ..Default::default()
    };
    let result = processor
        .apply_chat_template(&json_messages, params)
        .unwrap();

    // Check that the result contains expected markers
    assert!(result.contains("<|begin_of_text|>"));
    assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
    assert!(result.contains("You are a helpful assistant"));
    assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
    assert!(result.contains("What is 2+2?"));
    assert!(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
}

#[test]
fn test_chatml_template() {
    let template = r#"
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = [
        ChatMessage::User {
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        },
        ChatMessage::Assistant {
            content: Some(MessageContent::Text("Hi there!".to_string())),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        ChatMessage::User {
            content: MessageContent::Text("How are you?".to_string()),
            name: None,
        },
    ];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let result = processor
        .apply_chat_template(
            &json_messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .unwrap();

    // Check ChatML format
    assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
    assert!(result.contains("<|im_start|>assistant\nHi there!<|im_end|>"));
    assert!(result.contains("<|im_start|>user\nHow are you?<|im_end|>"));
    assert!(result.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_template_without_generation_prompt() {
    let template = r#"
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{% endfor -%}
{%- if add_generation_prompt -%}
assistant:
{%- endif -%}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = [ChatMessage::User {
        content: MessageContent::Text("Test".to_string()),
        name: None,
    }];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let result = processor
        .apply_chat_template(&json_messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(result.trim(), "user: Test");

    let result_with_prompt = processor
        .apply_chat_template(
            &json_messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(result_with_prompt.contains("assistant:"));
}

#[test]
fn test_empty_messages_template() {
    let template = r#"{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages: Vec<serde_json::Value> = vec![];
    let result = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_content_format_detection() {
    let string_template = r#"
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{%- endfor -%}
"#;
    assert_eq!(
        detect_chat_template_content_format(string_template),
        ChatTemplateContentFormat::String
    );

    let openai_template = r#"
{%- for message in messages -%}
  {%- for content in message.content -%}
    {{ content.type }}: {{ content.text }}
  {%- endfor -%}
{%- endfor -%}
"#;
    assert_eq!(
        detect_chat_template_content_format(openai_template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_template_with_multimodal_content() {
    let template = r#"
{%- for message in messages %}
{{ message.role }}:
{%- if message.content is string %}
{{ message.content }}
{%- else %}
{%- for part in message.content %}
  {%- if part.type == "text" %}
{{ part.text }}
  {%- elif part.type == "image_url" %}
[IMAGE]
  {%- endif %}
{%- endfor %}
{%- endif %}
{% endfor %}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = [ChatMessage::User {
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Look at this:".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            },
        ]),
        name: None,
    }];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let result = processor
        .apply_chat_template(&json_messages, ChatTemplateParams::default())
        .unwrap();

    // Should contain both text and image parts
    assert!(result.contains("user:"));
    assert!(result.contains("Look at this:"));
    assert!(result.contains("[IMAGE]"));
}
