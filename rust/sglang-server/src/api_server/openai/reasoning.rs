//! Reasoning-content splitting for Chat Completions (`--reasoning-parser`).
//!
//! Mirrors the Python frontend (`sglang.srt.parser.reasoning_parser` +
//! `serving_chat._process_reasoning_stream`): when the server was launched
//! with `--reasoning-parser <name>` (and the request keeps the default
//! `separate_reasoning=true`, which the Dynamo request type cannot express),
//! the model's `<think>`-style markers are stripped out of `content` into
//! `reasoning_content` — for unary responses and streaming deltas alike.

use dynamo_parsers::reasoning::{ReasoningParserType, ReasoningParserWrapper};

/// Build the parser the Python `--reasoning-parser` name selects.
///
/// The names come from Python's `ReasoningParser.DetectorMap`, which differs
/// from the dynamo-parsers registry keys in a few spellings (deepseek-r1 vs
/// deepseek_r1, kimi_k2 vs kimi_k25, …) and has a few entries that Python maps
/// onto a forced-reasoning `<think>` parser (qwen3-thinking, minimax). Names
/// dynamo does not know (hunyuan, inkling, apertus2509, mimo, poolside_v1,
/// cohere_command4 — all tokenizer-driven parsers) fall through to the
/// registry, which warns and falls back to the non-forced Basic parser.
pub(super) fn build_reasoning_parser(server_name: &str) -> ReasoningParserWrapper {
    let name = match server_name {
        // Python DetectorMap spellings that differ from the dynamo registry keys.
        "deepseek-r1" | "step3p5" => "deepseek_r1",
        "kimi_k2" => "kimi_k25",
        "gpt-oss" => "gpt_oss",
        "nemotron_3" => "nemotron3",
        "interns1" => "qwen3",
        // Python forces reasoning for these; the R1 parser is the same
        // `<think>` / `</think>` configuration with `force_reasoning=true`.
        "qwen3-thinking" | "minimax" => "deepseek_r1",
        _ => server_name,
    };
    ReasoningParserType::get_reasoning_parser_from_name(name)
}

#[cfg(test)]
mod tests {
    use super::build_reasoning_parser;
    use dynamo_parsers::reasoning::ReasoningParser;

    #[test]
    fn python_deepseek_r1_name_splits_forced_reasoning() {
        let mut parser = build_reasoning_parser("deepseek-r1");
        // Forced: text before any marker is reasoning.
        let split = parser.detect_and_parse_reasoning("think hard</think>Paris", &[]);
        assert_eq!(split.reasoning_text, "think hard");
        assert_eq!(split.normal_text, "Paris");
        let split = parser.detect_and_parse_reasoning("<think>yes</think>answer", &[]);
        assert_eq!(split.reasoning_text, "yes");
        assert_eq!(split.normal_text, "answer");
    }

    #[test]
    fn python_kimi_k2_name_maps_to_kimi_k25() {
        let mut parser = build_reasoning_parser("kimi_k2");
        let split = parser.detect_and_parse_reasoning("<think>k</think>out", &[]);
        assert_eq!(split.reasoning_text, "k");
        assert_eq!(split.normal_text, "out");
        // Kimi-K2.5 interrupts reasoning at the tool-call section marker.
        let mut parser = build_reasoning_parser("kimi_k2");
        let split =
            parser.detect_and_parse_reasoning("reasons<|tool_calls_section_begin|>calls", &[]);
        assert_eq!(split.reasoning_text, "reasons");
        assert_eq!(split.normal_text, "<|tool_calls_section_begin|>calls");
    }

    #[test]
    fn qwen3_thinking_forces_reasoning_like_python() {
        let mut parser = build_reasoning_parser("qwen3-thinking");
        let split = parser.detect_and_parse_reasoning("plain text", &[]);
        assert_eq!(split.reasoning_text, "plain text");
        assert_eq!(split.normal_text, "");
    }

    #[test]
    fn streaming_split_keeps_markers_out_of_both_columns() {
        let mut parser = build_reasoning_parser("deepseek-r1");
        let mut reasoning = String::new();
        let mut normal = String::new();
        for chunk in ["<think>rea", "son</think>an", "swer"] {
            let split = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&split.reasoning_text);
            normal.push_str(&split.normal_text);
        }
        let tail = parser.finish_reasoning_stream();
        reasoning.push_str(&tail.reasoning_text);
        normal.push_str(&tail.normal_text);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
    }
}
