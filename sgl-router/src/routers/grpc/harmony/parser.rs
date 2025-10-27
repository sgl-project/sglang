//! Harmony response parser
//!
//! Adapter for openai_harmony::StreamableParser that handles channel-based parsing.

use openai_harmony::{chat::Role, HarmonyEncoding, StreamableParser};
use uuid::Uuid;

use super::types::{HarmonyChannelDelta, HarmonyChannelOutput};
use crate::protocols::common::{FunctionCallResponse, ToolCall};

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
        let mut commentary: Option<Vec<ToolCall>> = None;
        let mut final_text = String::new();

        for msg in messages {
            // Filter: Only process assistant messages (vLLM lines 294-298)
            if msg.author.role != Role::Assistant {
                continue;
            }

            let channel = msg.channel.as_deref().unwrap_or("");
            let recipient = msg.recipient.as_deref();

            match channel {
                "analysis" => {
                    // vLLM lines 332-344: Process each content item
                    // For Chat API, we join them into a single reasoning_content
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    if !text.is_empty() {
                        analysis = Some(text);
                    }
                }
                "commentary" => {
                    // vLLM lines 345-377: Handle different recipient types
                    if let Some(recipient_str) = recipient {
                        if recipient_str.starts_with("functions.") {
                            // vLLM lines 346-357: Function tool calls
                            let function_name = recipient_str.strip_prefix("functions.").unwrap();

                            // Process each content item separately (vLLM line 348)
                            for content in &msg.content {
                                if let openai_harmony::chat::Content::Text(tc) = content {
                                    let call_id = format!("call_{}", Uuid::new_v4());
                                    let tool_call = ToolCall {
                                        id: call_id,
                                        tool_type: "function".to_string(),
                                        function: FunctionCallResponse {
                                            name: function_name.to_string(),
                                            arguments: Some(tc.text.clone()),
                                        },
                                    };

                                    match commentary.as_mut() {
                                        Some(calls) => calls.push(tool_call),
                                        None => commentary = Some(vec![tool_call]),
                                    }
                                }
                            }
                        } else if recipient_str.starts_with("python")
                            || recipient_str.starts_with("browser")
                            || recipient_str.starts_with("container")
                        {
                            // vLLM lines 358-375: Built-in tools → treat as reasoning
                            // For Chat API, we add to analysis content
                            let text = msg
                                .content
                                .iter()
                                .filter_map(|c| match c {
                                    openai_harmony::chat::Content::Text(tc) => {
                                        Some(tc.text.as_str())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("");

                            if !text.is_empty() {
                                // Append to analysis (built-in tools are reasoning)
                                match analysis.as_mut() {
                                    Some(existing) => {
                                        existing.push('\n');
                                        existing.push_str(&text);
                                    }
                                    None => analysis = Some(text),
                                }
                            }
                        }
                        // vLLM line 377: Unknown recipient would raise ValueError
                        // For now, we silently ignore (can add logging later)
                    }
                }
                "final" => {
                    // vLLM lines 378-395: Process final channel content
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    final_text.push_str(&text);
                }
                _ => {
                    // Unknown channel, append to final text as fallback
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

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
        let mut commentary: Option<Vec<ToolCall>> = None;
        let mut final_text = String::new();

        for msg in messages {
            // Filter: Only process assistant messages (vLLM lines 294-298)
            if msg.author.role != Role::Assistant {
                continue;
            }

            let channel = msg.channel.as_deref().unwrap_or("");
            let recipient = msg.recipient.as_deref();

            match channel {
                "analysis" => {
                    // vLLM lines 332-344: Process each content item
                    // For Chat API, we join them into a single reasoning_content
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    if !text.is_empty() {
                        analysis = Some(text);
                    }
                }
                "commentary" => {
                    // vLLM lines 345-377: Handle different recipient types
                    if let Some(recipient_str) = recipient {
                        if recipient_str.starts_with("functions.") {
                            // vLLM lines 346-357: Function tool calls
                            let function_name = recipient_str.strip_prefix("functions.").unwrap();

                            // Process each content item separately (vLLM line 348)
                            for content in &msg.content {
                                if let openai_harmony::chat::Content::Text(tc) = content {
                                    let call_id = format!("call_{}", Uuid::new_v4());
                                    let tool_call = ToolCall {
                                        id: call_id,
                                        tool_type: "function".to_string(),
                                        function: FunctionCallResponse {
                                            name: function_name.to_string(),
                                            arguments: Some(tc.text.clone()),
                                        },
                                    };

                                    match commentary.as_mut() {
                                        Some(calls) => calls.push(tool_call),
                                        None => commentary = Some(vec![tool_call]),
                                    }
                                }
                            }
                        } else if recipient_str.starts_with("python")
                            || recipient_str.starts_with("browser")
                            || recipient_str.starts_with("container")
                        {
                            // vLLM lines 358-375: Built-in tools → treat as reasoning
                            // For Chat API, we add to analysis content
                            let text = msg
                                .content
                                .iter()
                                .filter_map(|c| match c {
                                    openai_harmony::chat::Content::Text(tc) => {
                                        Some(tc.text.as_str())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("");

                            if !text.is_empty() {
                                // Append to analysis (built-in tools are reasoning)
                                match analysis.as_mut() {
                                    Some(existing) => {
                                        existing.push('\n');
                                        existing.push_str(&text);
                                    }
                                    None => analysis = Some(text),
                                }
                            }
                        }
                        // vLLM line 377: Unknown recipient would raise ValueError
                        // For now, we silently ignore (can add logging later)
                    }
                }
                "final" => {
                    // vLLM lines 378-395: Process final channel content
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    final_text.push_str(&text);
                }
                _ => {
                    // Unknown channel, append to final text as fallback
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            openai_harmony::chat::Content::Text(tc) => Some(tc.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");

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
