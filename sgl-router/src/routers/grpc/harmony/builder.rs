//! Harmony request builder
//!
//! Handles encoding of Chat/Responses requests into Harmony format using openai-harmony library.

use std::sync::OnceLock;

use openai_harmony::{
    chat::{Author, Content, Message as HarmonyMessage, Role, TextContent},
    HarmonyEncoding, HarmonyEncodingName,
};

use super::types::HarmonyBuildOutput;
use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, UserMessageContent};

/// Global Harmony encoding (lazy-initialized)
static HARMONY_ENCODING: OnceLock<HarmonyEncoding> = OnceLock::new();

/// Get or initialize the Harmony encoding
///
/// Uses HarmonyGptOss encoding which supports the gpt-oss model family.
pub(super) fn get_harmony_encoding() -> &'static HarmonyEncoding {
    HARMONY_ENCODING.get_or_init(|| {
        openai_harmony::load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .expect("Failed to load Harmony encoding")
    })
}

/// Harmony request builder
///
/// Converts OpenAI-format requests into Harmony-encoded format with input_ids,
/// stop tokens, and selection text for worker routing.
pub struct HarmonyBuilder {
    encoding: &'static HarmonyEncoding,
}

impl HarmonyBuilder {
    /// Create a new Harmony builder
    pub fn new() -> Self {
        Self {
            encoding: get_harmony_encoding(),
        }
    }

    /// Build Harmony request from Chat Completion request
    ///
    /// # Arguments
    ///
    /// * `request` - The ChatCompletionRequest to encode
    ///
    /// # Returns
    ///
    /// HarmonyBuildOutput containing input_ids, stop_token_ids, selection_text, and messages
    pub fn build_from_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<HarmonyBuildOutput, String> {
        // Step 1: Convert OpenAI messages to Harmony messages
        let harmony_messages = self.convert_chat_messages(&request.messages)?;

        // Step 2: Encode messages to token IDs using openai-harmony
        let mut token_ids = Vec::new();
        self.encoding
            .render_conversation_into(harmony_messages.iter(), &mut token_ids, None)
            .map_err(|e| format!("Failed to encode Harmony conversation: {}", e))?;

        // Step 3: Extract selection text from last user message (for worker routing)
        let selection_text = self.extract_selection_text(&harmony_messages);

        // Step 4: Stop token IDs (TODO: implement proper stop sequence encoding)
        // For now, we rely on the backend to handle stop sequences via the request's stop field
        let stop_token_ids = Vec::new();

        Ok(HarmonyBuildOutput {
            input_ids: token_ids,
            stop_token_ids,
            selection_text,
            harmony_messages: harmony_messages
                .into_iter()
                .map(|msg| super::types::HarmonyMessage::from_openai_harmony(msg))
                .collect(),
        })
    }

    /// Convert OpenAI ChatMessage format to Harmony messages
    fn convert_chat_messages(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<HarmonyMessage>, String> {
        let mut harmony_messages = Vec::with_capacity(messages.len());

        for msg in messages {
            let (harmony_role, text, msg_name) = match msg {
                ChatMessage::System { content, name } => {
                    (Role::System, content.clone(), name.clone())
                }
                ChatMessage::User { content, name } => {
                    let text = match content {
                        UserMessageContent::Text(text) => text.clone(),
                        UserMessageContent::Parts(parts) => {
                            // For multimodal content, extract text parts
                            // TODO: Handle images and other content types properly
                            parts
                                .iter()
                                .filter_map(|part| {
                                    if let crate::protocols::common::ContentPart::Text { text } =
                                        part
                                    {
                                        Some(text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    };
                    (Role::User, text, name.clone())
                }
                ChatMessage::Assistant {
                    content,
                    name,
                    reasoning_content,
                    ..
                } => {
                    // Combine content with reasoning if present
                    let mut text = content.clone().unwrap_or_default();
                    if let Some(reasoning) = reasoning_content {
                        if !text.is_empty() {
                            text.push_str("\n");
                        }
                        text.push_str(reasoning);
                    }
                    (Role::Assistant, text, name.clone())
                }
                ChatMessage::Tool {
                    content,
                    tool_call_id,
                } => {
                    // Tool messages become user messages with tool context
                    let text = format!("[Tool result for {}]: {}", tool_call_id, content);
                    (Role::User, text, None)
                }
                ChatMessage::Function { content, name } => {
                    // Function messages become user messages
                    let text = format!("[Function {name}]: {content}");
                    (Role::User, text, None)
                }
            };

            // Create Harmony message
            let mut harmony_msg = HarmonyMessage {
                author: Author {
                    role: harmony_role,
                    name: msg_name,
                },
                recipient: None,
                content: vec![Content::Text(TextContent { text })],
                channel: None,
                content_type: None,
            };

            // Set channel="final" for assistant messages
            if matches!(msg, ChatMessage::Assistant { .. }) {
                harmony_msg.channel = Some("final".to_string());
            }

            harmony_messages.push(harmony_msg);
        }

        Ok(harmony_messages)
    }

    /// Extract selection text for worker routing
    ///
    /// Uses the last user message (or a concise snippet) for load balancing
    fn extract_selection_text(&self, messages: &[HarmonyMessage]) -> String {
        // Find the last user message
        if let Some(last_user_msg) = messages.iter().rev().find(|m| m.author.role == Role::User) {
            // Extract text from content
            let text = last_user_msg
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::Text(tc) => Some(tc.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            // Return first 100 characters for routing
            return text.chars().take(100).collect();
        }

        // Fallback: concatenate all text and take first 100 chars
        messages
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::Text(tc) => Some(tc.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(100)
            .collect()
    }
}

impl Default for HarmonyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::chat::ChatMessage;

    #[test]
    fn test_builder_creation() {
        let builder = HarmonyBuilder::new();
        // Ensure it can be created
        let _ = builder;
    }

    #[test]
    fn test_convert_simple_messages() {
        use crate::protocols::chat::UserMessageContent;

        let builder = HarmonyBuilder::new();

        let messages = vec![
            ChatMessage::User {
                content: UserMessageContent::Text("Hello".to_string()),
                name: None,
            },
            ChatMessage::Assistant {
                content: Some("Hi there".to_string()),
                name: None,
                tool_calls: None,
                reasoning_content: None,
            },
        ];

        let harmony_messages = builder.convert_chat_messages(&messages).unwrap();
        assert_eq!(harmony_messages.len(), 2);
        assert_eq!(harmony_messages[0].author.role, Role::User);
        assert_eq!(harmony_messages[1].author.role, Role::Assistant);
        // Assistant message should have channel="final"
        assert_eq!(harmony_messages[1].channel.as_deref(), Some("final"));
    }

    #[test]
    fn test_extract_selection_text() {
        let builder = HarmonyBuilder::new();

        let messages = vec![
            HarmonyMessage {
                author: Author {
                    role: Role::User,
                    name: None,
                },
                recipient: None,
                content: vec![Content::Text(TextContent {
                    text: "Hello, how are you?".to_string(),
                })],
                channel: None,
                content_type: None,
            },
            HarmonyMessage {
                author: Author {
                    role: Role::Assistant,
                    name: None,
                },
                recipient: None,
                content: vec![Content::Text(TextContent {
                    text: "I'm doing well, thanks!".to_string(),
                })],
                channel: Some("final".to_string()),
                content_type: None,
            },
            HarmonyMessage {
                author: Author {
                    role: Role::User,
                    name: None,
                },
                recipient: None,
                content: vec![Content::Text(TextContent {
                    text: "Can you help me with something?".to_string(),
                })],
                channel: None,
                content_type: None,
            },
        ];

        let selection = builder.extract_selection_text(&messages);
        assert_eq!(selection, "Can you help me with something?");
    }

    #[test]
    fn test_extract_selection_text_truncation() {
        let builder = HarmonyBuilder::new();

        let long_message = "a".repeat(150);
        let messages = vec![HarmonyMessage {
            author: Author {
                role: Role::User,
                name: None,
            },
            recipient: None,
            content: vec![Content::Text(TextContent { text: long_message })],
            channel: None,
            content_type: None,
        }];

        let selection = builder.extract_selection_text(&messages);
        assert_eq!(selection.len(), 100);
        assert_eq!(selection, "a".repeat(100));
    }

    #[test]
    fn test_build_from_chat_encoding() {
        use crate::protocols::chat::UserMessageContent;

        let builder = HarmonyBuilder::new();

        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::User {
                content: UserMessageContent::Text("Hello".to_string()),
                name: None,
            }],
            model: "gpt-4o".to_string(),
            frequency_penalty: None,
            logit_bias: None,
            logprobs: false,
            max_completion_tokens: None,
            metadata: None,
            modalities: None,
            n: Some(1),
            parallel_tool_calls: None,
            presence_penalty: None,
            prompt_cache_key: None,
            reasoning_effort: None,
            response_format: None,
            safety_identifier: None,
            service_tier: None,
            stop: None,
            stream: false,
            stream_options: None,
            temperature: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_p: None,
            verbosity: None,
            top_k: None,
            min_p: None,
            min_tokens: None,
            repetition_penalty: None,
            regex: None,
            ebnf: None,
            stop_token_ids: None,
            no_stop_trim: false,
            ignore_eos: false,
            continue_final_message: false,
            skip_special_tokens: true,
            lora_path: None,
            session_params: None,
            separate_reasoning: true,
            stream_reasoning: true,
            chat_template_kwargs: None,
            return_hidden_states: false,
            sampling_seed: None,
            #[allow(deprecated)]
            function_call: None,
            #[allow(deprecated)]
            functions: None,
            #[allow(deprecated)]
            max_tokens: None,
            #[allow(deprecated)]
            seed: None,
        };

        let output = builder.build_from_chat(&request).unwrap();

        // Verify output has token_ids (actual encoding happened)
        assert!(!output.input_ids.is_empty(), "Expected non-empty token IDs");

        // Verify selection text was extracted
        assert_eq!(output.selection_text, "Hello");

        // Verify harmony messages were created
        assert_eq!(output.harmony_messages.len(), 1);
        assert_eq!(output.harmony_messages[0].role, "user");
        assert_eq!(output.harmony_messages[0].content, "Hello");
    }
}
