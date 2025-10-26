//! Harmony request builder
//!
//! Handles encoding of Chat/Responses requests into Harmony format using openai-harmony library.

use std::sync::OnceLock;

use chrono::Local;
use openai_harmony::{
    chat::{
        Author, ChannelConfig, Content, Conversation, DeveloperContent, Message as HarmonyMessage,
        ReasoningEffort, Role, SystemContent, TextContent, ToolDescription,
    },
    HarmonyEncoding, HarmonyEncodingName,
};

use super::types::HarmonyBuildOutput;
use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage, UserMessageContent},
    common::{ContentPart, Tool},
};

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
        let mut all_messages = Vec::new();

        let sys_msg = self.build_system_message_from_chat(request);
        all_messages.push(sys_msg);

        let dev_msg = self.get_developer_message(request.tools.as_ref());
        all_messages.push(dev_msg);

        let mut user_messages = self.convert_chat_messages(&request.messages)?;
        all_messages.append(&mut user_messages);

        let conversation = Conversation::from_messages(all_messages.clone());
        let token_ids = self
            .encoding
            .render_conversation_for_completion(&conversation, Role::Assistant, None)
            .map_err(|e| format!("Failed to encode Harmony conversation: {}", e))?;

        let selection_text = self.extract_selection_text(&all_messages);

        // TODO: implement proper stop sequence encoding
        let stop_token_ids = Vec::new();

        Ok(HarmonyBuildOutput {
            input_ids: token_ids,
            stop_token_ids,
            selection_text,
            harmony_messages: all_messages
                .into_iter()
                .map(super::types::HarmonyMessage::from_openai_harmony)
                .collect(),
        })
    }

    /// Build system message from ChatCompletionRequest
    ///
    /// Follows vLLM's get_system_message() logic for Chat Completion API
    fn build_system_message_from_chat(&self, request: &ChatCompletionRequest) -> HarmonyMessage {
        let mut sys_content = SystemContent::new();

        // Add reasoning_effort if provided
        if let Some(effort) = request.reasoning_effort.as_deref() {
            let effort_enum = match effort {
                "high" => ReasoningEffort::High,
                "medium" => ReasoningEffort::Medium,
                "low" => ReasoningEffort::Low,
                _ => ReasoningEffort::Medium,
            };
            sys_content = sys_content.with_reasoning_effort(effort_enum);
        }
        sys_content =
            sys_content.with_conversation_start_date(Local::now().format("%Y-%m-%d").to_string());

        // If no tools, remove "commentary" from valid channels
        if request.tools.is_none() {
            if let Some(channel_config) = &sys_content.channel_config {
                let valid_channels: Vec<String> = channel_config
                    .valid_channels
                    .iter()
                    .filter(|c| c.as_str() != "commentary")
                    .cloned()
                    .collect();
                sys_content = sys_content
                    .with_channel_config(ChannelConfig::require_channels(valid_channels));
            }
        }

        HarmonyMessage::from_role_and_content(Role::System, sys_content)
    }

    /// Create developer message with tool descriptions
    ///
    /// - Filters out built-in tools (web_search_preview, code_interpreter, container)
    /// - Extracts function tools and MCP tools, converts to ToolDescription
    /// - Adds function tools to DeveloperContent
    ///
    /// # Arguments
    ///
    /// * `tools` - Optional list of tools from the request
    ///
    /// # Returns
    ///
    /// Harmony Message with Role::Developer containing tool descriptions
    fn get_developer_message(&self, tools: Option<&Vec<Tool>>) -> HarmonyMessage {
        let mut dev_content = DeveloperContent::new();

        // Early return if no tools
        let Some(tools) = tools else {
            return HarmonyMessage::from_role_and_content(Role::Developer, dev_content);
        };

        let mut function_tools = Vec::new();

        for tool in tools {
            match tool.tool_type.as_str() {
                "web_search_preview" | "code_interpreter" | "container" => {
                    // These are built-in tools that are added to the system message.
                    // Skip them in the developer message.
                    continue;
                }
                "mcp" => {
                    // We support MCP tools (unlike vLLM which skips them for now).
                    // Add them as function tools.
                    function_tools.push(tool);
                }
                "function" => {
                    function_tools.push(tool);
                }
                _ => {
                    // Unknown tool type - skip it
                    continue;
                }
            }
        }

        // Convert function tools to ToolDescription and add to developer content
        if !function_tools.is_empty() {
            let tool_descriptions: Vec<ToolDescription> = function_tools
                .iter()
                .map(|tool| {
                    ToolDescription::new(
                        tool.function.name.clone(),
                        tool.function.description.clone().unwrap_or_default(),
                        Some(tool.function.parameters.clone()),
                    )
                })
                .collect();

            dev_content = dev_content.with_function_tools(tool_descriptions);
        }

        HarmonyMessage::from_role_and_content(Role::Developer, dev_content)
    }

    /// Convert OpenAI ChatMessage format to Harmony messages
    ///
    /// Following vLLM's parse_chat_input logic:
    /// - Assistant messages with tool_calls create multiple messages (one per tool call)
    /// - Tool role messages use Role::Tool with proper author
    /// - Tool-related messages use channel="commentary"
    fn convert_chat_messages(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<HarmonyMessage>, String> {
        let mut harmony_messages = Vec::new();

        // Build a map of tool_call_id -> function_name for tool responses
        let mut tool_call_map = std::collections::HashMap::new();
        for msg in messages {
            if let ChatMessage::Assistant {
                tool_calls: Some(calls),
                ..
            } = msg
            {
                for call in calls {
                    tool_call_map.insert(call.id.clone(), call.function.name.clone());
                }
            }
        }

        for msg in messages {
            match msg {
                ChatMessage::System { content, name } => {
                    // System messages stay as-is
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::System,
                            name: name.clone(),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent {
                            text: content.clone(),
                        })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }

                ChatMessage::User { content, name } => {
                    // Extract text from user content
                    let text = match content {
                        UserMessageContent::Text(text) => text.clone(),
                        UserMessageContent::Parts(parts) => {
                            // For multimodal content, extract text parts
                            parts
                                .iter()
                                .filter_map(|part| {
                                    if let ContentPart::Text { text } = part {
                                        Some(text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    };

                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::User,
                            name: name.clone(),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent { text })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }

                ChatMessage::Assistant {
                    content,
                    name,
                    tool_calls,
                    reasoning_content,
                } => {
                    // Following vLLM logic: if there are tool_calls, create separate messages
                    if let Some(calls) = tool_calls {
                        // Create one message per tool call with channel="commentary"
                        for call in calls {
                            let function_name = &call.function.name;
                            let arguments = call.function.arguments.clone().unwrap_or_default();

                            let tool_call_msg = HarmonyMessage {
                                author: Author {
                                    role: Role::Assistant,
                                    name: name.clone(),
                                },
                                recipient: Some(format!("functions.{}", function_name)),
                                content: vec![Content::Text(TextContent { text: arguments })],
                                channel: Some("commentary".to_string()),
                                content_type: Some("json".to_string()),
                            };
                            harmony_messages.push(tool_call_msg);
                        }
                    } else {
                        // Regular assistant message with content
                        // Combine content with reasoning if present
                        let mut text = content.clone().unwrap_or_default();
                        if let Some(reasoning) = reasoning_content {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(reasoning);
                        }

                        let harmony_msg = HarmonyMessage {
                            author: Author {
                                role: Role::Assistant,
                                name: name.clone(),
                            },
                            recipient: None,
                            content: vec![Content::Text(TextContent { text })],
                            channel: Some("final".to_string()),
                            content_type: None,
                        };
                        harmony_messages.push(harmony_msg);
                    }
                }

                ChatMessage::Tool {
                    content,
                    tool_call_id,
                } => {
                    // Following vLLM: Use Role::Tool with Author name "functions.{function_name}"
                    // Look up the function name from the tool_call_id
                    let function_name = tool_call_map
                        .get(tool_call_id)
                        .cloned()
                        .unwrap_or_else(|| tool_call_id.clone());

                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(format!("functions.{}", function_name)),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent {
                            text: content.clone(),
                        })],
                        channel: Some("commentary".to_string()),
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }

                ChatMessage::Function { content, name } => {
                    // Function messages also use Role::Tool
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(format!("functions.{}", name)),
                        },
                        recipient: None,
                        content: vec![Content::Text(TextContent {
                            text: content.clone(),
                        })],
                        channel: Some("commentary".to_string()),
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }
            }
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
    use crate::protocols::{
        chat::ChatMessage,
        common::{FunctionCallResponse, ToolCall},
    };

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
    fn test_convert_tool_calls() {
        use crate::protocols::chat::UserMessageContent;

        let builder = HarmonyBuilder::new();

        let messages = vec![
            ChatMessage::User {
                content: UserMessageContent::Text("What's the weather?".to_string()),
                name: None,
            },
            ChatMessage::Assistant {
                content: None,
                name: None,
                tool_calls: Some(vec![
                    ToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: "get_weather".to_string(),
                            arguments: Some(r#"{"location": "SF"}"#.to_string()),
                        },
                    },
                    ToolCall {
                        id: "call_2".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: "get_time".to_string(),
                            arguments: Some(r#"{"timezone": "PST"}"#.to_string()),
                        },
                    },
                ]),
                reasoning_content: None,
            },
        ];

        let harmony_messages = builder.convert_chat_messages(&messages).unwrap();
        // Should have 3 messages: 1 user + 2 tool calls
        assert_eq!(harmony_messages.len(), 3);

        // First message is user
        assert_eq!(harmony_messages[0].author.role, Role::User);

        // Second and third are tool calls
        assert_eq!(harmony_messages[1].author.role, Role::Assistant);
        assert_eq!(harmony_messages[1].channel.as_deref(), Some("commentary"));
        assert_eq!(
            harmony_messages[1].recipient.as_deref(),
            Some("functions.get_weather")
        );
        assert_eq!(harmony_messages[1].content_type.as_deref(), Some("json"));

        assert_eq!(harmony_messages[2].author.role, Role::Assistant);
        assert_eq!(harmony_messages[2].channel.as_deref(), Some("commentary"));
        assert_eq!(
            harmony_messages[2].recipient.as_deref(),
            Some("functions.get_time")
        );
    }

    #[test]
    fn test_convert_tool_response() {
        let builder = HarmonyBuilder::new();

        let messages = vec![ChatMessage::Tool {
            content: r#"{"temperature": 72}"#.to_string(),
            tool_call_id: "call_1".to_string(),
        }];

        let harmony_messages = builder.convert_chat_messages(&messages).unwrap();
        assert_eq!(harmony_messages.len(), 1);

        // Should use Role::Tool
        assert_eq!(harmony_messages[0].author.role, Role::Tool);
        assert_eq!(
            harmony_messages[0].author.name.as_deref(),
            Some("functions.call_1")
        );
        assert_eq!(harmony_messages[0].channel.as_deref(), Some("commentary"));
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

        // Verify harmony messages were created: system + developer + user
        assert_eq!(
            output.harmony_messages.len(),
            3,
            "Expected 3 messages: system, developer, user"
        );
        assert_eq!(output.harmony_messages[0].role, "system");
        assert_eq!(output.harmony_messages[1].role, "developer");
        assert_eq!(output.harmony_messages[2].role, "user");
        assert_eq!(output.harmony_messages[2].content, "Hello");
    }
}
