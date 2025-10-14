use sglang_router_rs::protocols::spec;
use sglang_router_rs::tokenizer::chat_template::{
    detect_chat_template_content_format, ChatTemplateContentFormat, ChatTemplateParams,
    ChatTemplateProcessor,
};

#[test]
fn test_detect_string_format_deepseek() {
    // DeepSeek style template - expects string content
    let template = r#"
        {%- for message in messages %}
        {%- if message['role'] == 'user' %}
        User: {{ message['content'] }}
        {%- elif message['role'] == 'assistant' %}
        Assistant: {{ message['content'] }}
        {%- endif %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::String
    );
}

#[test]
fn test_detect_openai_format_llama4() {
    // Llama4 style template - expects structured content
    let template = r#"
        {%- for message in messages %}
        {%- if message['content'] is iterable %}
        {%- for content in message['content'] %}
        {%- if content['type'] == 'text' %}
        {{ content['text'] }}
        {%- elif content['type'] == 'image' %}
        <image>
        {%- endif %}
        {%- endfor %}
        {%- else %}
        {{ message['content'] }}
        {%- endif %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_detect_openai_format_dot_notation() {
    // Template using dot notation
    let template = r#"
        {%- for message in messages %}
        {%- for part in message.content %}
        {%- if part.type == 'text' %}
        {{ part.text }}
        {%- endif %}
        {%- endfor %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_detect_openai_format_variable_assignment() {
    // Template that assigns content to variable then iterates
    let template = r#"
        {%- for message in messages %}
        {%- set content = message['content'] %}
        {%- if content is sequence %}
        {%- for item in content %}
        {{ item }}
        {%- endfor %}
        {%- endif %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_detect_openai_format_glm4v_style() {
    // GLM4V uses 'msg' instead of 'message'
    let template = r#"
        {%- for msg in messages %}
        {%- for part in msg.content %}
        {%- if part.type == 'text' %}{{ part.text }}{%- endif %}
        {%- if part.type == 'image' %}<image>{%- endif %}
        {%- endfor %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_detect_openai_format_with_length_check() {
    // Template that checks content length
    let template = r#"
        {%- for message in messages %}
        {%- if message.content|length > 0 %}
        {%- for item in message.content %}
        {{ item.text }}
        {%- endfor %}
        {%- endif %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_detect_openai_format_with_index_access() {
    // Template that accesses content by index
    let template = r#"
        {%- for message in messages %}
        {%- if message.content[0] %}
        First item: {{ message.content[0].text }}
        {%- endif %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_invalid_template_defaults_to_string() {
    let template = "Not a valid {% jinja template";

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::String
    );
}

#[test]
fn test_empty_template_defaults_to_string() {
    assert_eq!(
        detect_chat_template_content_format(""),
        ChatTemplateContentFormat::String
    );
}

#[test]
fn test_simple_chat_template_unit_test() {
    let template = r#"
{%- for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor -%}
{%- if add_generation_prompt %}
assistant:
{%- endif %}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = vec![
        spec::ChatMessage::System {
            role: "system".to_string(),
            content: "You are helpful".to_string(),
            name: None,
        },
        spec::ChatMessage::User {
            role: "user".to_string(),
            content: spec::UserMessageContent::Text("Hello".to_string()),
            name: None,
        },
    ];

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
    assert!(result.contains("system: You are helpful"));
    assert!(result.contains("user: Hello"));
    assert!(result.contains("assistant:"));
}

#[test]
fn test_chat_template_with_tokens_unit_test() {
    // Template that uses template kwargs for tokens (more realistic)
    let template = r#"
{%- if start_token -%}{{ start_token }}{%- endif -%}
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}{%- if end_token -%}{{ end_token }}{%- endif -%}
{% endfor -%}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string());

    let messages = [spec::ChatMessage::User {
        role: "user".to_string(),
        content: spec::UserMessageContent::Text("Test".to_string()),
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
        "start_token".to_string(),
        serde_json::Value::String("<s>".to_string()),
    );
    template_kwargs.insert(
        "end_token".to_string(),
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
