// MiniMax M2 specific reasoning parser.
// This parser automatically appends <think> token at the beginning of text,
// similar to the Python MiniMaxAppendThinkDetector.

use crate::reasoning_parser::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser},
};

/// MiniMax M2 reasoning parser.
///
/// This parser automatically appends <think> token at the beginning of the first chunk
/// and uses <think> and </think> tokens for reasoning blocks.
pub struct MiniMaxParser {
    base: BaseReasoningParser,
    is_first_chunk: bool,
}

impl MiniMaxParser {
    /// Create a new MiniMax M2 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: 65536,
            initial_in_reasoning: false, // Start with false, we'll add <think> manually
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("minimax".to_string()),
            is_first_chunk: true,
        }
    }
}

impl Default for MiniMaxParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MiniMaxParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        // For one-shot parsing, prepend <think> token to the text
        let modified_text = format!("<think>{}", text);
        self.base.detect_and_parse_reasoning(&modified_text)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        // For the first chunk, prepend <think> token
        let modified_text = if self.is_first_chunk {
            self.is_first_chunk = false;
            format!("<think>{}", text)
        } else {
            text.to_string()
        };

        self.base
            .parse_reasoning_streaming_incremental(&modified_text)
    }

    fn reset(&mut self) {
        self.base.reset();
        self.is_first_chunk = true; // Reset the first chunk flag
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
    }

    fn is_in_reasoning(&self) -> bool {
        self.base.is_in_reasoning()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_append_think_oneshot() {
        let mut parser = MiniMaxParser::new();

        // Should automatically prepend <think> and parse as reasoning
        let result = parser
            .detect_and_parse_reasoning("reasoning content</think>normal content")
            .unwrap();
        assert_eq!(result.normal_text, "normal content");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_minimax_without_end_token() {
        let mut parser = MiniMaxParser::new();

        // Should treat all content as reasoning when no end token
        let result = parser
            .detect_and_parse_reasoning("all reasoning content")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "all reasoning content");
    }

    #[test]
    fn test_minimax_streaming_first_chunk() {
        let mut parser = MiniMaxParser::new();

        // First chunk should have <think> prepended
        let result1 = parser
            .parse_reasoning_streaming_incremental("thinking about")
            .unwrap();
        assert_eq!(result1.reasoning_text, "thinking about");
        assert_eq!(result1.normal_text, "");

        // Second chunk should not have <think> prepended
        let result2 = parser
            .parse_reasoning_streaming_incremental(" the problem</think>answer")
            .unwrap();
        assert_eq!(result2.reasoning_text, "the problem"); // Text is trimmed
        assert_eq!(result2.normal_text, "answer");
    }

    #[test]
    fn test_minimax_reset() {
        let mut parser = MiniMaxParser::new();

        // First use
        let result1 = parser
            .parse_reasoning_streaming_incremental("first")
            .unwrap();
        assert_eq!(result1.reasoning_text, "first");

        // Reset the parser
        parser.reset();

        // After reset, should be first chunk again
        let result2 = parser
            .parse_reasoning_streaming_incremental("second")
            .unwrap();
        assert_eq!(result2.reasoning_text, "second");
    }

    #[test]
    fn test_minimax_already_has_think() {
        let mut parser = MiniMaxParser::new();

        // Even if text already has <think>, it will add another one
        // This mimics the Python behavior
        let result = parser
            .detect_and_parse_reasoning("<think>content</think>answer")
            .unwrap();
        // The double <think> gets handled by the base parser which removes duplicates
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "content");
    }

    #[test]
    fn test_model_type() {
        let parser = MiniMaxParser::new();
        assert_eq!(parser.model_type(), "minimax");
    }
}
