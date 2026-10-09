//! Reasoning-content splitting for `--reasoning-parser`.

use dynamo_parsers::reasoning::{
    ReasoningParser as _, ReasoningParserType, ReasoningParserWrapper,
};

/// Build the parser a Python `--reasoning-parser` name selects. Names Dynamo
/// does not know fall back to its non-forced basic parser.
fn build_reasoning_parser(server_name: &str) -> ReasoningParserWrapper {
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

fn u32_ids<T: Copy + TryInto<u32>>(ids: &[T]) -> Vec<u32> {
    ids.iter().filter_map(|&id| id.try_into().ok()).collect()
}

/// Split a finished generation into `(reasoning_text, normal_text)`. Without
/// a parser the text is all normal.
pub fn split_reasoning<T: Copy + TryInto<u32>>(
    name: Option<&str>,
    text: &str,
    token_ids: &[T],
) -> (String, String) {
    let Some(name) = name else {
        return (String::new(), text.to_owned());
    };
    let split = build_reasoning_parser(name).detect_and_parse_reasoning(text, &u32_ids(token_ids));
    (split.reasoning_text, split.normal_text)
}

/// Stateful reasoning split for one streamed choice. The parser is built on
/// the first frame; `initial_reasoning` overrides its starting state, e.g.
/// when the prompt already opened `<think>`.
pub struct ReasoningStreamSplitter {
    name: Option<String>,
    parser: Option<ReasoningParserWrapper>,
    pub(super) initial_reasoning: Option<bool>,
}

impl ReasoningStreamSplitter {
    pub fn new(name: Option<&str>, initial_reasoning: Option<bool>) -> Self {
        Self {
            name: name.map(str::to_owned),
            parser: None,
            initial_reasoning,
        }
    }

    /// Split one frame's text into `(reasoning_text, normal_text)` deltas.
    pub fn split<T: Copy + TryInto<u32>>(
        &mut self,
        text: &str,
        token_ids: &[T],
    ) -> (String, String) {
        let Some(name) = self.name.as_deref() else {
            return (String::new(), text.to_owned());
        };
        let initial_reasoning = self.initial_reasoning;
        let parser = self.parser.get_or_insert_with(|| {
            let mut parser = build_reasoning_parser(name);
            if let Some(initial_reasoning) = initial_reasoning {
                parser.set_in_reasoning(initial_reasoning);
            }
            parser
        });
        let split = parser.parse_reasoning_streaming_incremental(text, &u32_ids(token_ids));
        (split.reasoning_text, split.normal_text)
    }

    /// Flush the buffered tail at stream end; it can sit in either column.
    pub fn finish(&mut self) -> (String, String) {
        let Some(parser) = self.parser.as_mut() else {
            return (String::new(), String::new());
        };
        let tail = parser.finish_reasoning_stream();
        (tail.reasoning_text, tail.normal_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_IDS: &[u32] = &[];

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

    #[test]
    fn unary_split_passes_text_through_without_a_parser() {
        let text = "<think>kept as text</think>";
        assert_eq!(
            split_reasoning(None, text, &[1i64]),
            (String::new(), text.into())
        );
    }

    #[test]
    fn v4_streaming_separates_prefilled_reasoning() {
        let mut splitter = ReasoningStreamSplitter::new(Some("deepseek-v4"), Some(true));
        assert_eq!(
            splitter.split("reason", NO_IDS),
            ("reason".into(), "".into())
        );
        assert_eq!(
            splitter.split("</think>answer", NO_IDS),
            ("".into(), "answer".into())
        );
    }

    #[test]
    fn streaming_split_keeps_markers_out_of_both_columns() {
        let mut splitter = ReasoningStreamSplitter::new(Some("deepseek-r1"), None);
        let mut columns = (String::new(), String::new());
        for chunk in ["<think>rea", "son</think>an", "swer"] {
            let (reasoning, normal) = splitter.split(chunk, NO_IDS);
            columns.0 += &reasoning;
            columns.1 += &normal;
        }
        let (reasoning, normal) = splitter.finish();
        assert_eq!(
            (columns.0 + &reasoning, columns.1 + &normal),
            ("reason".into(), "answer".into())
        );
        let mut plain = ReasoningStreamSplitter::new(None, None);
        assert_eq!(plain.split("plain", NO_IDS), ("".into(), "plain".into()));
        assert_eq!(plain.finish(), ("".into(), "".into()));
    }

    #[test]
    fn minimax_m3_tail_lands_in_the_right_column() {
        // M3 holds an ambiguous prefix until a boundary or the end.
        let mut splitter = ReasoningStreamSplitter::new(Some("minimax_m3"), None);
        assert_eq!(
            splitter.split("The answer is", NO_IDS),
            ("".into(), "".into())
        );
        assert_eq!(splitter.split(" 42", NO_IDS), ("".into(), "".into()));
        assert_eq!(splitter.finish(), ("".into(), "The answer is 42".into()));

        let mut splitter = ReasoningStreamSplitter::new(Some("minimax_m3"), None);
        assert_eq!(
            splitter.split("<mm:think>think", NO_IDS),
            ("think".into(), "".into())
        );
        assert_eq!(
            splitter.split(" hard</mm:think>", NO_IDS),
            (" hard".into(), "".into())
        );
        assert_eq!(splitter.finish(), ("".into(), "".into()));
    }
}
