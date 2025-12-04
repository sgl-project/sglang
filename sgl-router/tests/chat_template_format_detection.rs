use sglang_router_rs::{
    protocols::chat::{ChatMessage, MessageContent},
    tokenizer::chat_template::{
        detect_chat_template_content_format, ChatTemplateContentFormat, ChatTemplateParams,
        ChatTemplateProcessor,
    },
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

    let messages = [
        ChatMessage::System {
            content: MessageContent::Text("You are helpful".to_string()),
            name: None,
        },
        ChatMessage::User {
            content: MessageContent::Text("Hello".to_string()),
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

#[test]
fn test_detect_openai_format_qwen3vl_macro_style() {
    // Qwen3-VL style template using macros to handle multimodal content
    // This tests the macro-based detection pattern
    let template = r#"{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count) %}
    {%- if content is string %}
        {{- content }}
    {%- else %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}
                <|vision_start|><|image_pad|><|vision_end|>
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}
                <|vision_start|><|video_pad|><|vision_end|>
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endmacro %}
{%- for message in messages %}
    {%- set content = render_content(message.content, True) %}
    {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_detect_openai_format_arbitrary_variable_names() {
    // Test that detection works with any variable name, not just "message", "msg", "m"
    // Uses "chat_msg" and "x" as loop variables
    let template = r#"
        {%- for chat_msg in messages %}
        {%- for x in chat_msg.content %}
        {%- if x.type == 'text' %}{{ x.text }}{%- endif %}
        {%- if x.type == 'image' %}<image>{%- endif %}
        {%- endfor %}
        {%- endfor %}
        "#;

    assert_eq!(
        detect_chat_template_content_format(template),
        ChatTemplateContentFormat::OpenAI
    );
}
