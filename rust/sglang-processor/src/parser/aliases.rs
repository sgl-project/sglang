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

pub(super) fn build_reasoning_parser(server_name: &str) -> ReasoningParserWrapper {
    let name = match server_name {
        "deepseek-r1" | "step3p5" => "deepseek_r1",
        "kimi_k2" => "kimi_k25",
        "gpt-oss" => "gpt_oss",
        "nemotron_3" => "nemotron3",
        "interns1" => "qwen3",
        "qwen3-thinking" | "minimax" => "deepseek_r1",
        _ => server_name,
    };
    ReasoningParserType::get_reasoning_parser_from_name(name)
}
