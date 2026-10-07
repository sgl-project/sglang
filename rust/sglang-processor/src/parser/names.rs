//! SGLang parser names mapped onto Dynamo's.

use dynamo_parsers::reasoning::{ReasoningParserType, ReasoningParserWrapper};

/// Map SGLang tool-parser aliases onto Dynamo's tool-parser names.
pub fn dynamo_tool_parser_name(parser: &str) -> &str {
    match parser {
        "llama3" => "llama3_json",
        "qwen" => "qwen25",
        "glm" | "glm45" => "glm47",
        other => other,
    }
}

/// Build the parser a Python `--reasoning-parser` name selects. Names Dynamo
/// does not know fall back to its non-forced basic parser.
pub(super) fn build_reasoning_parser(server_name: &str) -> ReasoningParserWrapper {
    let name = match server_name {
        "deepseek-r1" | "step3p5" => "deepseek_r1",
        "kimi_k2" => "kimi_k25",
        "gpt-oss" => "gpt_oss",
        "nemotron_3" => "nemotron3",
        "interns1" => "qwen3",
        // Python forces reasoning for these; R1 is the same `<think>` parser, forced.
        "qwen3-thinking" | "minimax" => "deepseek_r1",
        _ => server_name,
    };
    ReasoningParserType::get_reasoning_parser_from_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers::reasoning::ReasoningParser as _;

    #[test]
    fn tool_parser_aliases() {
        assert_eq!(dynamo_tool_parser_name("llama3"), "llama3_json");
        assert_eq!(dynamo_tool_parser_name("qwen"), "qwen25");
        assert_eq!(dynamo_tool_parser_name("glm45"), "glm47");
        assert_eq!(dynamo_tool_parser_name("deepseekv4"), "deepseekv4");
    }

    #[test]
    fn reasoning_parser_aliases_keep_python_semantics() {
        let split = build_reasoning_parser("deepseek-r1")
            .detect_and_parse_reasoning("think hard</think>Paris", &[]);
        assert_eq!(
            (split.reasoning_text.as_str(), split.normal_text.as_str()),
            ("think hard", "Paris")
        );
        let split = build_reasoning_parser("kimi_k2")
            .detect_and_parse_reasoning("reasons<|tool_calls_section_begin|>calls", &[]);
        assert_eq!(split.reasoning_text, "reasons");
        let split =
            build_reasoning_parser("qwen3-thinking").detect_and_parse_reasoning("plain", &[]);
        assert_eq!(split.reasoning_text, "plain");
    }
}
