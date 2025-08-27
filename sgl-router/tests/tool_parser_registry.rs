//! Parser Registry Integration Tests
//!
//! Tests for model-to-parser mappings and registry functionality

use sglang_router_rs::tool_parser::ParserRegistry;

#[tokio::test]
async fn test_registry_has_all_parsers() {
    let registry = ParserRegistry::new();
    let parsers = registry.list_parsers();

    assert!(parsers.contains(&"json"));
    assert!(parsers.contains(&"mistral"));
    assert!(parsers.contains(&"qwen"));
    assert!(parsers.contains(&"pythonic"));
    assert!(parsers.contains(&"llama"));
}

#[tokio::test]
async fn test_openai_models_use_json() {
    let registry = ParserRegistry::new();

    let models = vec!["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"];
    for model in models {
        let parser = registry.get_parser(model).unwrap();
        let test_input = r#"{"name": "test", "arguments": {}}"#;
        let result = parser.parse_complete(test_input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "test");
    }
}

#[tokio::test]
async fn test_anthropic_models_use_json() {
    let registry = ParserRegistry::new();

    let models = vec!["claude-3-opus", "claude-3-sonnet", "claude-2.1"];
    for model in models {
        let parser = registry.get_parser(model).unwrap();
        let test_input = r#"{"name": "test", "arguments": {}}"#;
        let result = parser.parse_complete(test_input).await.unwrap();
        assert_eq!(result.len(), 1);
    }
}

#[tokio::test]
async fn test_mistral_models() {
    let registry = ParserRegistry::new();

    let models = vec!["mistral-large", "mistral-medium", "mixtral-8x7b"];
    for model in models {
        let parser = registry.get_parser(model).unwrap();
        let test_input = r#"[TOOL_CALLS] [{"name": "test", "arguments": {}}]"#;
        let result = parser.parse_complete(test_input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "test");
    }
}

#[tokio::test]
async fn test_qwen_models() {
    let registry = ParserRegistry::new();

    let models = vec!["qwen2.5-72b", "Qwen2-7B", "qwen-max"];
    for model in models {
        let parser = registry.get_parser(model).unwrap();
        let test_input = r#"<tool_call>
{"name": "test", "arguments": {}}
</tool_call>"#;
        let result = parser.parse_complete(test_input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "test");
    }
}

#[tokio::test]
async fn test_llama_model_variants() {
    let registry = ParserRegistry::new();

    // Llama 4 uses pythonic
    let parser = registry.get_parser("llama-4-70b").unwrap();
    let test_input = r#"[get_weather(city="NYC")]"#;
    let result = parser.parse_complete(test_input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");

    // Llama 3.2 uses python_tag
    let parser = registry.get_parser("llama-3.2-8b").unwrap();
    let test_input = r#"<|python_tag|>{"name": "test", "arguments": {}}"#;
    let result = parser.parse_complete(test_input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");

    // Other Llama models use JSON
    let parser = registry.get_parser("llama-2-70b").unwrap();
    let test_input = r#"{"name": "test", "arguments": {}}"#;
    let result = parser.parse_complete(test_input).await.unwrap();
    assert_eq!(result.len(), 1);
}

#[tokio::test]
async fn test_deepseek_models() {
    let registry = ParserRegistry::new();

    // DeepSeek uses pythonic format (simplified, v3 would need custom parser)
    let parser = registry.get_parser("deepseek-coder").unwrap();
    let test_input = r#"[function(arg="value")]"#;
    let result = parser.parse_complete(test_input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "function");
}

#[tokio::test]
async fn test_unknown_model_fallback() {
    let registry = ParserRegistry::new();

    // Unknown models should fall back to JSON parser
    let parser = registry.get_parser("unknown-model-xyz").unwrap();
    let test_input = r#"{"name": "fallback", "arguments": {}}"#;
    let result = parser.parse_complete(test_input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "fallback");
}

#[tokio::test]
async fn test_pattern_specificity() {
    let registry = ParserRegistry::new();

    // Test that more specific patterns take precedence
    // llama-4* should match before llama-*
    let parser = registry.get_parser("llama-4-70b").unwrap();
    assert!(parser.detect_format(r#"[test_function(x=1)]"#)); // Pythonic format

    let parser = registry.get_parser("llama-3-70b").unwrap();
    assert!(parser.detect_format(r#"{"name": "test", "arguments": {}}"#)); // JSON format
}

#[tokio::test]
async fn test_real_world_model_outputs() {
    let registry = ParserRegistry::new();

    // Test with realistic outputs from different models
    let test_cases = vec![
        (
            "gpt-4",
            r#"I'll help you with that.

{"name": "search_web", "arguments": {"query": "latest AI news", "max_results": 5}}

Let me search for that information."#,
            "search_web",
        ),
        (
            "mistral-large",
            r#"Let me search for information about Rust.

[TOOL_CALLS] [
    {"name": "search", "arguments": {"query": "Rust programming"}},
    {"name": "get_weather", "arguments": {"city": "San Francisco"}}
]

I've initiated the search."#,
            "search",
        ),
        (
            "qwen2.5",
            r#"I'll check the weather for you.

<tool_call>
{
    "name": "get_weather",
    "arguments": {
        "location": "Tokyo",
        "units": "celsius"
    }
}
</tool_call>

The weather information has been requested."#,
            "get_weather",
        ),
    ];

    for (model, output, expected_name) in test_cases {
        let parser = registry.get_parser(model).unwrap();
        let result = parser.parse_complete(output).await.unwrap();
        assert!(!result.is_empty(), "No tools parsed for model {}", model);
        assert_eq!(
            result[0].function.name, expected_name,
            "Wrong function name for model {}",
            model
        );
    }
}
