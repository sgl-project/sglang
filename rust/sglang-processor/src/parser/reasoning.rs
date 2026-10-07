//! Reasoning-content splitting for `--reasoning-parser`.

use dynamo_parsers::reasoning::{ReasoningParser as _, ReasoningParserWrapper};

use super::names::build_reasoning_parser;

/// Split a finished generation into `(reasoning_text, normal_text)`. Without
/// a parser the text is all normal.
pub fn split_reasoning(name: Option<&str>, text: &str, token_ids: &[u32]) -> (String, String) {
    let Some(name) = name else {
        return (String::new(), text.to_owned());
    };
    let split = build_reasoning_parser(name).detect_and_parse_reasoning(text, token_ids);
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
    pub fn split(&mut self, text: &str, token_ids: &[u32]) -> (String, String) {
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
        let split = parser.parse_reasoning_streaming_incremental(text, token_ids);
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

    #[test]
    fn unary_split_passes_text_through_without_a_parser() {
        let text = "<think>kept as text</think>";
        assert_eq!(
            split_reasoning(None, text, &[1]),
            (String::new(), text.into())
        );
    }

    #[test]
    fn v4_streaming_separates_prefilled_reasoning() {
        let mut splitter = ReasoningStreamSplitter::new(Some("deepseek-v4"), Some(true));
        assert_eq!(splitter.split("reason", &[]), ("reason".into(), "".into()));
        assert_eq!(
            splitter.split("</think>answer", &[]),
            ("".into(), "answer".into())
        );
    }

    #[test]
    fn streaming_split_keeps_markers_out_of_both_columns() {
        let mut splitter = ReasoningStreamSplitter::new(Some("deepseek-r1"), None);
        let mut columns = (String::new(), String::new());
        for chunk in ["<think>rea", "son</think>an", "swer"] {
            let (reasoning, normal) = splitter.split(chunk, &[]);
            columns.0 += &reasoning;
            columns.1 += &normal;
        }
        let (reasoning, normal) = splitter.finish();
        assert_eq!(
            (columns.0 + &reasoning, columns.1 + &normal),
            ("reason".into(), "answer".into())
        );
        let mut plain = ReasoningStreamSplitter::new(None, None);
        assert_eq!(plain.split("plain", &[]), ("".into(), "plain".into()));
        assert_eq!(plain.finish(), ("".into(), "".into()));
    }

    #[test]
    fn streaming_tail_releases_normal_text_only_at_finish() {
        // MiniMax M3 holds an ambiguous prefix until a boundary or the end.
        let mut splitter = ReasoningStreamSplitter::new(Some("minimax_m3"), None);
        assert_eq!(splitter.split("The answer is", &[]), ("".into(), "".into()));
        assert_eq!(splitter.split(" 42", &[]), ("".into(), "".into()));
        assert_eq!(splitter.finish(), ("".into(), "The answer is 42".into()));
    }
}
