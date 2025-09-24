#[cfg(test)]
mod tests {
    use sglang_router_rs::tokenizer::chat_template::{ChatMessage, ChatTemplateProcessor};

    #[test]
    fn test_chat_message_helpers() {
        let system_msg = ChatMessage::system("You are a helpful assistant");
        assert_eq!(system_msg.role, "system");
        assert_eq!(system_msg.content, "You are a helpful assistant");

        let user_msg = ChatMessage::user("Hello!");
        assert_eq!(user_msg.role, "user");
        assert_eq!(user_msg.content, "Hello!");

        let assistant_msg = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant_msg.role, "assistant");
        assert_eq!(assistant_msg.content, "Hi there!");
    }

    #[test]
    fn test_llama_style_template() {
        // Test a Llama-style chat template
        let template = r#"
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
{%- endif -%}

{{- bos_token }}
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

        let processor = ChatTemplateProcessor::new(
            template.to_string(),
            Some("<|begin_of_text|>".to_string()),
            Some("<|end_of_text|>".to_string()),
        );

        let messages = vec![
            ChatMessage::system("You are a helpful assistant"),
            ChatMessage::user("What is 2+2?"),
        ];

        let result = processor.apply_chat_template(&messages, true).unwrap();

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
        // Test a ChatML-style template
        let template = r#"
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

        let processor = ChatTemplateProcessor::new(template.to_string(), None, None);

        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ];

        let result = processor.apply_chat_template(&messages, true).unwrap();

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

        let processor = ChatTemplateProcessor::new(template.to_string(), None, None);

        let messages = vec![ChatMessage::user("Test")];

        // Test without generation prompt
        let result = processor.apply_chat_template(&messages, false).unwrap();
        assert_eq!(result.trim(), "user: Test");

        // Test with generation prompt
        let result_with_prompt = processor.apply_chat_template(&messages, true).unwrap();
        assert!(result_with_prompt.contains("assistant:"));
    }

    #[test]
    fn test_template_with_special_tokens() {
        let template = r#"{{ bos_token }}{% for msg in messages %}{{ msg.content }}{{ eos_token }}{% endfor %}"#;

        let processor = ChatTemplateProcessor::new(
            template.to_string(),
            Some("<s>".to_string()),
            Some("</s>".to_string()),
        );

        let messages = vec![ChatMessage::user("Hello")];

        let result = processor.apply_chat_template(&messages, false).unwrap();
        assert_eq!(result, "<s>Hello</s>");
    }

    #[test]
    fn test_empty_messages() {
        let template =
            r#"{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}"#;

        let processor = ChatTemplateProcessor::new(template.to_string(), None, None);

        let messages = vec![];
        let result = processor.apply_chat_template(&messages, false).unwrap();
        assert_eq!(result, "");
    }

    // Integration test with actual tokenizer file loading would go here
    // but requires a real tokenizer_config.json file
}
