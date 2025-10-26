//! Harmony response parser
//!
//! Adapter for openai_harmony::StreamableParser that handles channel-based parsing.

use openai_harmony::{chat::Role, HarmonyEncoding, StreamableParser};

use super::types::{HarmonyChannelDelta, HarmonyChannelOutput};

/// Get the global Harmony encoding
///
/// References the same encoding used by the builder for consistency
fn get_harmony_encoding() -> &'static HarmonyEncoding {
    use super::builder::get_harmony_encoding;
    get_harmony_encoding()
}

/// Harmony parser adapter
///
/// Wraps openai_harmony::StreamableParser and provides methods for parsing
/// complete responses and streaming chunks.
pub struct HarmonyParserAdapter {
    parser: StreamableParser,
}

impl HarmonyParserAdapter {
    /// Create a new Harmony parser
    pub fn new() -> Result<Self, String> {
        let encoding = get_harmony_encoding();
        let parser = StreamableParser::new(encoding.clone(), Some(Role::Assistant))
            .map_err(|e| format!("Failed to create StreamableParser: {}", e))?;

        Ok(Self { parser })
    }

    /// Parse complete response
    ///
    /// Parses all output token IDs and returns the complete channel output
    /// containing analysis, commentary (tool calls), and final text.
    ///
    /// # Arguments
    ///
    /// * `output_ids` - The complete output token IDs from the model
    ///
    /// # Returns
    ///
    /// Complete HarmonyChannelOutput with all three channels parsed
    pub fn parse_complete(&mut self, output_ids: &[u32]) -> Result<HarmonyChannelOutput, String> {
        // Feed all tokens to the parser
        for &token_id in output_ids {
            self.parser
                .process(token_id)
                .map_err(|e| format!("Failed to process token {}: {}", token_id, e))?;
        }

        // Extract all completed messages from the parser
        let messages = self.parser.messages();

        // Parse messages into channel outputs
        let mut analysis = None;
        let mut commentary = None;
        let mut final_text = String::new();

        for msg in messages {
            let channel = msg.channel.as_deref().unwrap_or("");

            // Extract text content
            let text = msg
                .content
                .iter()
                .filter_map(|c| match c {
                    openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            match channel {
                "analysis" => {
                    analysis = Some(text);
                }
                "commentary" => {
                    // TODO: Parse tool calls from commentary channel
                    // For now, just store as text
                    commentary = None; // Placeholder
                }
                "final" => {
                    final_text.push_str(&text);
                }
                _ => {
                    // Unknown channel, append to final text as fallback
                    final_text.push_str(&text);
                }
            }
        }

        // Check for incomplete content in parser state
        if let Ok(current_content) = self.parser.current_content() {
            if !current_content.is_empty() {
                let current_channel = self.parser.current_channel();
                match current_channel.as_deref() {
                    Some("analysis") => {
                        analysis = Some(current_content);
                    }
                    Some("final") | None => {
                        final_text.push_str(&current_content);
                    }
                    _ => {}
                }
            }
        }

        Ok(HarmonyChannelOutput {
            analysis,
            commentary,
            final_text,
            finish_reason: "stop".to_string(), // TODO: Determine actual finish reason
            matched_stop: None,
        })
    }

    /// Parse streaming chunk
    ///
    /// Parses incremental token IDs and returns a delta with any new content
    /// from the analysis, commentary, or final channels.
    ///
    /// # Arguments
    ///
    /// * `chunk_ids` - New token IDs from the current chunk
    ///
    /// # Returns
    ///
    /// Optional HarmonyChannelDelta if there's new content to emit
    pub fn parse_chunk(
        &mut self,
        chunk_ids: &[u32],
    ) -> Result<Option<HarmonyChannelDelta>, String> {
        let mut has_delta = false;
        let mut analysis_delta = None;
        let mut final_delta = None;

        // Track message count before processing
        let prev_message_count = self.parser.messages().len();

        // Process each token
        for &token_id in chunk_ids {
            self.parser
                .process(token_id)
                .map_err(|e| format!("Failed to process token {}: {}", token_id, e))?;

            // Check for content delta
            if let Ok(Some(delta_text)) = self.parser.last_content_delta() {
                has_delta = true;

                // Determine which channel this delta belongs to
                let channel = self.parser.current_channel();
                match channel.as_deref() {
                    Some("analysis") => {
                        analysis_delta = Some(delta_text);
                    }
                    Some("final") | None => {
                        final_delta = Some(delta_text);
                    }
                    Some("commentary") => {
                        // TODO: Parse tool call deltas
                        // For now, skip commentary deltas
                    }
                    _ => {}
                }
            }
        }

        // Check if new messages were completed
        let current_message_count = self.parser.messages().len();
        let is_final = current_message_count > prev_message_count;

        if has_delta {
            Ok(Some(HarmonyChannelDelta {
                analysis_delta,
                commentary_delta: None, // TODO: Implement tool call delta parsing
                final_delta,
                is_final,
            }))
        } else {
            Ok(None)
        }
    }

    /// Finalize parsing
    ///
    /// Called at the end of streaming to get the final state and any
    /// remaining content.
    ///
    /// # Returns
    ///
    /// Final HarmonyChannelOutput with complete parsed content
    pub fn finalize(&mut self) -> Result<HarmonyChannelOutput, String> {
        // Extract all completed messages
        let messages = self.parser.messages();

        let mut analysis = None;
        let mut commentary = None;
        let mut final_text = String::new();

        for msg in messages {
            let channel = msg.channel.as_deref().unwrap_or("");

            let text = msg
                .content
                .iter()
                .filter_map(|c| match c {
                    openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            match channel {
                "analysis" => {
                    analysis = Some(text);
                }
                "commentary" => {
                    // TODO: Parse tool calls
                    commentary = None;
                }
                "final" => {
                    final_text.push_str(&text);
                }
                _ => {
                    final_text.push_str(&text);
                }
            }
        }

        // Check for remaining incomplete content
        if let Ok(current_content) = self.parser.current_content() {
            if !current_content.is_empty() {
                let current_channel = self.parser.current_channel();
                match current_channel.as_deref() {
                    Some("analysis") => {
                        analysis = Some(current_content);
                    }
                    Some("final") | None => {
                        final_text.push_str(&current_content);
                    }
                    _ => {}
                }
            }
        }

        Ok(HarmonyChannelOutput {
            analysis,
            commentary,
            final_text,
            finish_reason: "stop".to_string(),
            matched_stop: None,
        })
    }

    /// Reset parser state
    ///
    /// Resets the parser to initial state for reuse
    pub fn reset(&mut self) -> Result<(), String> {
        // Create a new parser instance (StreamableParser doesn't have a reset method)
        let encoding = get_harmony_encoding();
        self.parser = StreamableParser::new(encoding.clone(), Some(Role::Assistant))
            .map_err(|e| format!("Failed to reset parser: {}", e))?;
        Ok(())
    }
}

impl Default for HarmonyParserAdapter {
    fn default() -> Self {
        Self::new().expect("Failed to create default parser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = HarmonyParserAdapter::new();
        assert!(parser.is_ok(), "Parser creation should succeed");
    }

    #[test]
    fn test_parse_complete_empty() {
        let mut parser = HarmonyParserAdapter::new().unwrap();

        // Parse empty token list
        let output = parser.parse_complete(&[]).unwrap();
        assert_eq!(output.finish_reason, "stop");
        assert_eq!(output.final_text, "");
    }

    #[test]
    fn test_parse_chunk_empty() {
        let mut parser = HarmonyParserAdapter::new().unwrap();

        // Parse empty chunk
        let delta = parser.parse_chunk(&[]).unwrap();
        assert!(delta.is_none(), "Empty chunk should return None");
    }

    #[test]
    fn test_finalize() {
        let mut parser = HarmonyParserAdapter::new().unwrap();

        // Finalize without any tokens
        let output = parser.finalize().unwrap();
        assert_eq!(output.finish_reason, "stop");
    }

    #[test]
    fn test_reset() {
        let mut parser = HarmonyParserAdapter::new().unwrap();

        // Reset should succeed
        let result = parser.reset();
        assert!(result.is_ok(), "Reset should succeed");
    }

    #[test]
    fn test_parse_complete_with_tokens() {
        let mut parser = HarmonyParserAdapter::new().unwrap();

        // Test with some simple tokens (these are just examples, actual encoding is complex)
        // In real usage, tokens would come from encoding actual text
        let test_tokens = vec![100, 200, 300]; // Placeholder tokens

        // This might fail or return empty content depending on token validity
        // But it should not panic
        let result = parser.parse_complete(&test_tokens);
        assert!(result.is_ok(), "Parser should handle token sequence");
    }

    #[test]
    fn test_parse_chunk_incremental() {
        let mut parser = HarmonyParserAdapter::new().unwrap();

        // Parse chunks incrementally
        let chunk1 = vec![100, 200];
        let chunk2 = vec![300, 400];

        let delta1 = parser.parse_chunk(&chunk1);
        assert!(delta1.is_ok(), "First chunk should parse");

        let delta2 = parser.parse_chunk(&chunk2);
        assert!(delta2.is_ok(), "Second chunk should parse");
    }
}
