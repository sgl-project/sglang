//! Tool-parser aliases shared by request constraints and response parsing.

/// Map SGLang tool-parser aliases onto Dynamo's tool-parser names.
pub fn dynamo_tool_parser_name(parser: &str) -> &str {
    match parser {
        "llama3" => "llama3_json",
        "qwen" => "qwen25",
        "glm" | "glm45" => "glm47",
        other => other,
    }
}
