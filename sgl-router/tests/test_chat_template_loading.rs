#[cfg(test)]
mod tests {
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_chat_template_from_file() {
        use sglang_router_rs::tokenizer::chat_template::ChatMessage;
        use sglang_router_rs::tokenizer::huggingface::HuggingFaceTokenizer;

        // Create temporary directory
        let temp_dir = TempDir::new().unwrap();
        let template_path = temp_dir.path().join("template.jinja");

        // Write a test template
        let template_content = r#"
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>' + message['content'] }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>' }}
{%- endif %}
"#;
        fs::write(&template_path, template_content).unwrap();

        // Create a mock tokenizer config
        let tokenizer_config = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "<s>": 2,
                    "</s>": 3
                },
                "merges": []
            }
        }"#;

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        fs::write(&tokenizer_path, tokenizer_config).unwrap();

        // Load tokenizer with custom chat template
        let tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
            tokenizer_path.to_str().unwrap(),
            Some(template_path.to_str().unwrap()),
        )
        .unwrap();

        // Test that the custom template is used
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
        ];

        let result = tokenizer.apply_chat_template(&messages, true).unwrap();

        // Verify the custom template format
        assert!(result.contains("<|user|>Hello"));
        assert!(result.contains("<|assistant|>Hi there"));
        assert!(result.ends_with("<|assistant|>"));
    }

    #[test]
    fn test_override_existing_template() {
        use sglang_router_rs::tokenizer::chat_template::ChatMessage;
        use sglang_router_rs::tokenizer::huggingface::HuggingFaceTokenizer;

        // Create temporary directory
        let temp_dir = TempDir::new().unwrap();

        // Create tokenizer config with a built-in template
        let tokenizer_config_path = temp_dir.path().join("tokenizer_config.json");
        let config_with_template = r#"{
            "chat_template": "built-in: {% for msg in messages %}{{ msg.content }}{% endfor %}"
        }"#;
        fs::write(&tokenizer_config_path, config_with_template).unwrap();

        // Create the actual tokenizer file
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {
                    "test": 0,
                    "<s>": 1,
                    "</s>": 2
                },
                "merges": []
            }
        }"#;
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        fs::write(&tokenizer_path, tokenizer_json).unwrap();

        // Create custom template that should override
        let custom_template_path = temp_dir.path().join("custom.jinja");
        let custom_template =
            r#"CUSTOM: {% for msg in messages %}[{{ msg.role }}]: {{ msg.content }}{% endfor %}"#;
        fs::write(&custom_template_path, custom_template).unwrap();

        // Load with custom template - should override the built-in one
        let tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
            tokenizer_path.to_str().unwrap(),
            Some(custom_template_path.to_str().unwrap()),
        )
        .unwrap();

        let messages = vec![ChatMessage::user("Test")];
        let result = tokenizer.apply_chat_template(&messages, false).unwrap();

        // Should use CUSTOM template, not built-in
        assert!(result.starts_with("CUSTOM:"));
        assert!(result.contains("[user]: Test"));
        assert!(!result.contains("built-in:"));
    }

    #[test]
    fn test_set_chat_template_after_creation() {
        use sglang_router_rs::tokenizer::chat_template::ChatMessage;
        use sglang_router_rs::tokenizer::huggingface::HuggingFaceTokenizer;

        // Create temporary directory and tokenizer file
        let temp_dir = TempDir::new().unwrap();
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {
                    "test": 0,
                    "<s>": 1,
                    "</s>": 2
                },
                "merges": []
            }
        }"#;
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        fs::write(&tokenizer_path, tokenizer_json).unwrap();

        // Load tokenizer without custom template
        let mut tokenizer =
            HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();

        // Set a template after creation (mimics Python's behavior)
        let new_template =
            "NEW: {% for msg in messages %}{{ msg.role }}: {{ msg.content }}; {% endfor %}";
        tokenizer.set_chat_template(new_template.to_string());

        let messages = vec![ChatMessage::user("Hello"), ChatMessage::assistant("World")];
        let result = tokenizer.apply_chat_template(&messages, false).unwrap();

        assert!(result.starts_with("NEW:"));
        assert!(result.contains("user: Hello;"));
        assert!(result.contains("assistant: World;"));
    }
}
