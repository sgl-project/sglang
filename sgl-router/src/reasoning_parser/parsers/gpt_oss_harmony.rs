// GPT-OSS Harmony reasoning parser.
//
// This parser uses OpenAI's Harmony framework to parse token-level structured
// conversations with separate channels for reasoning (analysis) and normal text (final).

use std::sync::OnceLock;

use openai_harmony::{
    chat::{Content, Message, Role},
    load_harmony_encoding, HarmonyEncoding, HarmonyEncodingName, StreamableParser,
};

use crate::reasoning_parser::traits::{ParseError, ParserResult, ReasoningParser};

/// Global Harmony GPT-OSS encoding (initialized once, thread-safe).
static GLOBAL_HARMONY_GPTOSS_ENCODING: OnceLock<HarmonyEncoding> = OnceLock::new();

/// Get or initialize the global Harmony GPT-OSS encoding.
fn get_harmony_encoding() -> Result<&'static HarmonyEncoding, ParseError> {
    GLOBAL_HARMONY_GPTOSS_ENCODING.get_or_init(|| {
        match load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss) {
            Ok(enc) => enc,
            Err(e) => panic!("Failed to load Harmony encoding: {}", e),
        }
    });

    GLOBAL_HARMONY_GPTOSS_ENCODING
        .get()
        .ok_or_else(|| ParseError::ConfigError("Failed to get Harmony encoding".to_string()))
}

/// Extract text content from a Harmony message.
fn extract_text_from_message(message: &Message) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            Content::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// GPT-OSS Harmony reasoning parser.
///
/// This parser uses token-level parsing with Harmony's StreamableParser to separate
/// reasoning content (analysis channel) from normal text (final channel).
pub struct GptOssHarmonyReasoningParser {
    parser: StreamableParser,
    encoding: &'static HarmonyEncoding,
    model_type: String,
}

impl GptOssHarmonyReasoningParser {
    /// Create a new GPT-OSS Harmony reasoning parser.
    pub fn new() -> Result<Self, ParseError> {
        let encoding = get_harmony_encoding()?;
        let parser = StreamableParser::new(encoding.clone(), Some(Role::Assistant))
            .map_err(|e| ParseError::ConfigError(format!("Failed to create parser: {}", e)))?;

        Ok(Self {
            parser,
            encoding,
            model_type: "gpt-oss-harmony".to_string(),
        })
    }

    /// Extract reasoning and normal text from parsed messages.
    fn extract_content(&self, messages: &[Message]) -> ParserResult {
        let mut reasoning_text = String::new();
        let mut normal_text = String::new();

        for message in messages {
            // Only process assistant messages
            if message.author.role != Role::Assistant {
                continue;
            }

            match message.channel.as_deref() {
                Some("analysis") => {
                    // Reasoning content
                    let text = extract_text_from_message(message);
                    if !text.is_empty() {
                        reasoning_text.push_str(&text);
                    }
                }
                Some("final") | None => {
                    // Normal text content
                    let text = extract_text_from_message(message);
                    if !text.is_empty() {
                        normal_text.push_str(&text);
                    }
                }
                Some("commentary") => {
                    // Tool call content - skip for reasoning parser
                }
                _ => {
                    // Unknown channel - treat as normal text
                    let text = extract_text_from_message(message);
                    if !text.is_empty() {
                        normal_text.push_str(&text);
                    }
                }
            }
        }

        ParserResult::new(normal_text, reasoning_text)
    }
}

impl Default for GptOssHarmonyReasoningParser {
    fn default() -> Self {
        Self::new().expect("Failed to create default GptOssHarmonyReasoningParser")
    }
}

impl ReasoningParser for GptOssHarmonyReasoningParser {
    // Text-based methods not supported - this is a token-only parser
    fn detect_and_parse_reasoning(&mut self, _text: &str) -> Result<ParserResult, ParseError> {
        // Not supported - use detect_and_parse_reasoning_from_tokens instead
        Ok(ParserResult::default())
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        _text: &str,
    ) -> Result<ParserResult, ParseError> {
        // Not supported - use parse_reasoning_streaming_incremental_from_tokens instead
        Ok(ParserResult::default())
    }

    // Token-based methods - the actual implementation
    fn detect_and_parse_reasoning_from_tokens(
        &mut self,
        token_ids: &[u32],
    ) -> Result<ParserResult, ParseError> {
        // Create a fresh parser for this parse operation
        // (StreamableParser doesn't have reset, so we recreate it)
        self.parser = StreamableParser::new(self.encoding.clone(), Some(Role::Assistant))
            .map_err(|e| ParseError::ConfigError(format!("Failed to create parser: {}", e)))?;

        // Process each token through Harmony's StreamableParser
        for &token_id in token_ids {
            self.parser
                .process(token_id)
                .map_err(|e| ParseError::ConfigError(format!("Failed to process token: {}", e)))?;
        }

        // Extract all parsed messages
        let messages = self.parser.messages();

        // Handle different message counts
        match messages.len() {
            0 => {
                // No complete messages yet - check if there's partial content
                let current_content = self.parser.current_content().map_err(|e| {
                    ParseError::ConfigError(format!("Failed to get current content: {}", e))
                })?;

                // Incomplete parsing - treat as reasoning for now
                Ok(ParserResult::reasoning(current_content))
            }
            _ => {
                // One or more complete messages - extract reasoning and normal text
                Ok(self.extract_content(messages))
            }
        }
    }

    fn parse_reasoning_streaming_incremental_from_tokens(
        &mut self,
        token_ids: &[u32],
    ) -> Result<ParserResult, ParseError> {
        let mut normal_delta = String::new();
        let mut reasoning_delta = String::new();

        // Process each token and extract deltas
        for &token_id in token_ids {
            self.parser
                .process(token_id)
                .map_err(|e| ParseError::ConfigError(format!("Failed to process token: {}", e)))?;

            // Get delta from last token
            let delta_result = self.parser.last_content_delta().map_err(|e| {
                ParseError::ConfigError(format!("Failed to get content delta: {}", e))
            })?;

            if let Some(delta) = delta_result {
                if let Some(channel) = self.parser.current_channel() {
                    match channel.as_str() {
                        "final" => normal_delta.push_str(&delta),
                        "analysis" => reasoning_delta.push_str(&delta),
                        "commentary" => {
                            // Tool call content - for tool parser, not reasoning parser
                            // Note: In a full implementation, we might want to recover
                            // the consumed metadata tokens here for tool parsing
                        }
                        _ => {
                            // Unknown channel - treat as normal text
                            normal_delta.push_str(&delta);
                        }
                    }
                }
            }
        }

        Ok(ParserResult::new(normal_delta, reasoning_delta))
    }

    fn supports_token_parsing(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        // Recreate the parser since StreamableParser doesn't have a reset method
        self.parser = StreamableParser::new(self.encoding.clone(), Some(Role::Assistant))
            .expect("Failed to recreate parser in reset");
    }

    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn is_in_reasoning(&self) -> bool {
        // Check if current channel is "analysis"
        self.parser
            .current_channel()
            .map(|ch| ch.as_str() == "analysis")
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = GptOssHarmonyReasoningParser::new();
        assert!(parser.is_ok());
    }

    #[test]
    fn test_model_type() {
        let parser = GptOssHarmonyReasoningParser::new().unwrap();
        assert_eq!(parser.model_type(), "gpt-oss-harmony");
    }

    #[test]
    fn test_reset() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();
        parser.reset();
        // Parser should be in initial state after reset
        assert!(!parser.is_in_reasoning());
    }

    #[test]
    fn test_initial_state() {
        let parser = GptOssHarmonyReasoningParser::new().unwrap();
        assert!(!parser.is_in_reasoning());
        assert_eq!(parser.model_type(), "gpt-oss-harmony");
    }

    // Helper function to encode text to tokens for testing
    fn encode_text(text: &str) -> Vec<u32> {
        let encoding = get_harmony_encoding().expect("Failed to get encoding");
        encoding
            .tokenizer()
            .encode_ordinary(text)
            .into_iter()
            .collect()
    }

    #[test]
    fn test_simple_final_channel_message() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Encode a simple message in the final channel
        // Format: <|start|>assistant<|message|>Hello world
        let start_tokens = encode_text("<|start|>assistant<|message|>Hello world");

        let result = parser.detect_and_parse_reasoning_from_tokens(&start_tokens);

        // For incomplete message (no stop token), we expect it to be treated as reasoning
        // until we have a complete parsed message
        assert!(result.is_ok());
    }

    #[test]
    fn test_analysis_channel_detection() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Test with analysis channel marker
        let tokens = encode_text("<|start|>assistant<|channel|>analysis<|message|>thinking");

        let result = parser.detect_and_parse_reasoning_from_tokens(&tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_final_channel_detection() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Test with final channel marker
        let tokens = encode_text("<|start|>assistant<|channel|>final<|message|>answer");

        let result = parser.detect_and_parse_reasoning_from_tokens(&tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_token_list() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Should handle empty token list gracefully
        let result = parser.detect_and_parse_reasoning("");
        assert!(result.is_ok());

        let parsed = result.unwrap();
        // With no tokens, should have no content
        assert_eq!(parsed.normal_text, "");
    }

    #[test]
    fn test_streaming_incremental_empty() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Empty streaming should work
        let result = parser.parse_reasoning_streaming_incremental("");
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.normal_text, "");
        assert_eq!(parsed.reasoning_text, "");
    }

    #[test]
    fn test_streaming_incremental_basic() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Test basic streaming with some tokens
        let tokens = encode_text("test");

        let result = parser.parse_reasoning_streaming_incremental_from_tokens(&tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_resets() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Should handle multiple resets
        parser.reset();
        assert!(!parser.is_in_reasoning());

        parser.reset();
        assert!(!parser.is_in_reasoning());

        parser.reset();
        assert!(!parser.is_in_reasoning());
    }

    #[test]
    fn test_token_based_parsing() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // This is a token-only parser - use token-based methods
        let tokens = encode_text("actual content");

        // Use the token-based method
        let result = parser.detect_and_parse_reasoning_from_tokens(&tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoding_singleton() {
        // Encoding should be initialized once and reused
        let enc1 = get_harmony_encoding();
        let enc2 = get_harmony_encoding();

        assert!(enc1.is_ok());
        assert!(enc2.is_ok());

        // Should be same reference (OnceLock ensures single initialization)
        assert_eq!(enc1.unwrap() as *const _, enc2.unwrap() as *const _);
    }

    #[test]
    fn test_channel_extraction_logic() {
        let parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Test that channel detection works with is_in_reasoning
        assert!(!parser.is_in_reasoning()); // Initial state
    }

    #[test]
    fn test_default_trait() {
        // Test Default trait implementation
        let parser = GptOssHarmonyReasoningParser::default();
        assert_eq!(parser.model_type(), "gpt-oss-harmony");
        assert!(!parser.is_in_reasoning());
    }

    #[test]
    fn test_concurrent_parsing() {
        // Test that parser state is properly managed
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        let tokens1 = encode_text("first");
        let result1 = parser.detect_and_parse_reasoning_from_tokens(&tokens1);
        assert!(result1.is_ok());

        // Reset before second parse
        parser.reset();

        let tokens2 = encode_text("second");
        let result2 = parser.detect_and_parse_reasoning_from_tokens(&tokens2);
        assert!(result2.is_ok());
    }

    #[test]
    fn test_large_token_sequence() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Test with a larger sequence
        let large_text =
            "This is a longer piece of text that will generate more tokens for testing purposes.";
        let tokens = encode_text(large_text);

        assert!(tokens.len() > 10); // Should have multiple tokens

        let result = parser.detect_and_parse_reasoning_from_tokens(&tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_streaming_state_persistence() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Test that streaming maintains state across calls
        let tokens1 = encode_text("first part");
        let result1 = parser.parse_reasoning_streaming_incremental_from_tokens(&tokens1);
        assert!(result1.is_ok());

        let tokens2 = encode_text(" second part");
        let result2 = parser.parse_reasoning_streaming_incremental_from_tokens(&tokens2);
        assert!(result2.is_ok());

        // State should accumulate
    }

    #[test]
    fn test_special_tokens_handling() {
        let mut parser = GptOssHarmonyReasoningParser::new().unwrap();

        // Test with Harmony formatting tokens
        let tokens = encode_text("<|start|><|message|><|end|>");

        let result = parser.detect_and_parse_reasoning_from_tokens(&tokens);
        assert!(result.is_ok());
    }
}
