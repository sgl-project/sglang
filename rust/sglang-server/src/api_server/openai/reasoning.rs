//! Reasoning-content splitting for Chat Completions (`--reasoning-parser`).
//!
//! Mirrors the Python frontend (`sglang.srt.parser.reasoning_parser` +
//! `serving_chat._process_reasoning_stream`): when the
//! server was launched with `--reasoning-parser <name>` (and the request keeps
//! the default `separate_reasoning=true`, which the Dynamo request type cannot
//! express), the model's `<think>`-style markers are stripped out of `content`
//! into `reasoning_content` — for unary responses and streaming deltas alike.
//!
//! The parser lifecycle (lazy build, per-frame incremental split, terminal
//! flush of *both* buffered columns) lives here so the endpoint cannot drop
//! the tail half.

use dynamo_parsers::reasoning::{
    ReasoningParser as _, ReasoningParserType, ReasoningParserWrapper,
};

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

/// Split a completed generation's text into `(reasoning_text, normal_text)`
/// when `--reasoning-parser` selects a parser; otherwise the text passes
/// through untouched as normal text. Chat splits before tool-call parsing.
pub(super) fn split_reasoning_unary(
    name: Option<&str>,
    text: &str,
    token_ids: &[i32],
) -> (String, String) {
    let Some(name) = name else {
        return (String::new(), text.to_owned());
    };
    let mut parser = build_reasoning_parser(name);
    let token_ids = token_ids
        .iter()
        .filter_map(|&id| u32::try_from(id).ok())
        .collect::<Vec<_>>();
    let split = parser.detect_and_parse_reasoning(text, &token_ids);
    (split.reasoning_text, split.normal_text)
}

/// Stateful reasoning split for one streaming response. Mirrors Python's
/// `reasoning_parser_dict` entries: the parser is built lazily on the first
/// content delta, each frame is split into `(reasoning, normal)` deltas, and
/// [`finish`](Self::finish) flushes the parser-buffered tail — *both* columns,
/// since the buffered text can sit in either one (e.g. MiniMax M3's
/// implicit-tool-start recovery holds the leading answer text until the think
/// opener or a tool marker establishes the mode, and releases it as normal
/// text at EOF).
#[derive(Default)]
pub(super) struct ReasoningStreamSplitter {
    name: Option<String>,
    parser: Option<ReasoningParserWrapper>,
}

impl ReasoningStreamSplitter {
    pub(super) fn new(name: Option<&str>) -> Self {
        Self {
            name: name.map(str::to_owned),
            parser: None,
        }
    }

    /// Split one frame's text into `(reasoning_text, normal_text)` deltas.
    pub(super) fn split(&mut self, text: &str, token_ids: &[i32]) -> (String, String) {
        let Some(name) = self.name.as_deref() else {
            return (String::new(), text.to_owned());
        };
        let parser = self
            .parser
            .get_or_insert_with(|| build_reasoning_parser(name));
        let token_ids = token_ids
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect::<Vec<_>>();
        let split = parser.parse_reasoning_streaming_incremental(text, &token_ids);
        (split.reasoning_text, split.normal_text)
    }

    /// Flush the parser-buffered tail at stream end, releasing both columns.
    pub(super) fn finish(&mut self) -> (String, String) {
        let Some(parser) = self.parser.as_mut() else {
            return (String::new(), String::new());
        };
        let tail = parser.finish_reasoning_stream();
        (tail.reasoning_text, tail.normal_text)
    }
}

#[cfg(test)]
mod tests {
    use super::{ReasoningStreamSplitter, build_reasoning_parser, split_reasoning_unary};
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

    #[test]
    fn unary_split_passes_text_through_without_a_parser() {
        let (reasoning, normal) = split_reasoning_unary(None, "<think>kept as text</think>", &[1]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "<think>kept as text</think>");
    }

    /// REASONING_P1: MiniMax M3's implicit-tool-start recovery buffers the
    /// answer text until a boundary establishes the mode; with no opener the
    /// whole buffer is released as normal text only at `finish`. The chat
    /// terminal flush must emit the normal half of the tail.
    #[test]
    fn streaming_tail_releases_normal_text_only_at_finish() {
        let mut splitter = ReasoningStreamSplitter::new(Some("minimax_m3"));
        let (reasoning, normal) = splitter.split("The answer is", &[]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "", "M3 holds the ambiguous prefix until a boundary");
        let (reasoning, normal) = splitter.split(" 42", &[]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "");
        let (reasoning, normal) = splitter.finish();
        assert_eq!(reasoning, "");
        assert_eq!(normal, "The answer is 42");
    }

    #[test]
    fn streaming_tail_releases_reasoning_after_marker_boundary() {
        let mut splitter = ReasoningStreamSplitter::new(Some("minimax_m3"));
        let (reasoning, normal) = splitter.split("<mm:think>think", &[]);
        assert_eq!(reasoning, "think");
        assert_eq!(normal, "");
        let (reasoning, normal) = splitter.split(" hard</mm:think>", &[]);
        assert_eq!(reasoning, " hard");
        assert_eq!(normal, "");
        let (reasoning, normal) = splitter.finish();
        assert_eq!(reasoning, "");
        assert_eq!(normal, "");
    }

    #[test]
    fn finish_without_a_parser_is_empty() {
        let mut splitter = ReasoningStreamSplitter::new(None);
        let (reasoning, normal) = splitter.split("plain", &[]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "plain");
        let (reasoning, normal) = splitter.finish();
        assert_eq!(reasoning, "");
        assert_eq!(normal, "");
    }
}
