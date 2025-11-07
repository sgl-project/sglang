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
use tracing::debug;

use super::types::HarmonyBuildOutput;
use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage, UserMessageContent},
    common::{ContentPart, Tool},
    responses::{
        ReasoningEffort as ResponsesReasoningEffort, ResponseContentPart, ResponseInput,
        ResponseInputOutputItem, ResponseReasoningContent, ResponseTool, ResponseToolType,
        ResponsesRequest, StringOrContentParts,
    },
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

/// Built-in tools that are added to the system message
const BUILTIN_TOOLS: &[&str] = &["web_search_preview", "code_interpreter", "container"];

/// Trait for tool-like objects that can be converted to Harmony ToolDescription
trait ToolLike {
    /// Check if this is a built-in tool (should be skipped in developer message)
    #[allow(dead_code)]
    fn is_builtin(&self) -> bool;

    /// Check if this is a custom tool (function or MCP)
    fn is_custom(&self) -> bool;

    /// Convert to ToolDescription
    fn to_tool_description(&self) -> Option<ToolDescription>;
}

/// Implement ToolLike for Chat Completion Tool
impl ToolLike for Tool {
    fn is_builtin(&self) -> bool {
        matches!(
            self.tool_type.as_str(),
            "web_search_preview" | "code_interpreter" | "container"
        )
    }

    fn is_custom(&self) -> bool {
        matches!(self.tool_type.as_str(), "mcp" | "function")
    }

    fn to_tool_description(&self) -> Option<ToolDescription> {
        Some(ToolDescription::new(
            self.function.name.clone(),
            self.function.description.clone().unwrap_or_default(),
            Some(self.function.parameters.clone()),
        ))
    }
}

/// Implement ToolLike for Responses API Tool
impl ToolLike for ResponseTool {
    fn is_builtin(&self) -> bool {
        matches!(
            self.r#type,
            ResponseToolType::WebSearchPreview | ResponseToolType::CodeInterpreter
        )
    }

    fn is_custom(&self) -> bool {
        matches!(
            self.r#type,
            ResponseToolType::Mcp | ResponseToolType::Function
        )
    }

    fn to_tool_description(&self) -> Option<ToolDescription> {
        self.function.as_ref().map(|func| {
            ToolDescription::new(
                func.name.clone(),
                func.description.clone().unwrap_or_default(),
                Some(func.parameters.clone()),
            )
        })
    }
}

fn has_custom_tools(tool_types: &[&str]) -> bool {
    !tool_types.iter().all(|t| BUILTIN_TOOLS.contains(t))
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

        let dev_msg = self.build_developer_message_from_chat(request.tools.as_ref());
        all_messages.push(dev_msg);

        let mut user_messages = self.convert_chat_messages(&request.messages)?;
        all_messages.append(&mut user_messages);

        let conversation = Conversation::from_messages(all_messages.clone());
        let token_ids = self
            .encoding
            .render_conversation_for_completion(&conversation, Role::Assistant, None)
            .map_err(|e| format!("Failed to encode Harmony conversation: {}", e))?;

        let selection_text = self.extract_selection_text(&all_messages);

        // Get stop tokens for Harmony assistant actions (<|return|> and <|call|>)
        let stop_token_ids: Vec<u32> = self
            .encoding
            .stop_tokens_for_assistant_actions()
            .into_iter()
            .flat_map(|set| set.into_iter())
            .collect();

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

    /// Build Harmony request from Responses request
    ///
    /// # Arguments
    ///
    /// * `request` - The ResponsesRequest to encode
    ///
    /// # Returns
    ///
    /// HarmonyBuildOutput containing input_ids, stop_token_ids, selection_text, and messages
    pub fn build_from_responses(
        &self,
        request: &ResponsesRequest,
    ) -> Result<HarmonyBuildOutput, String> {
        let all_messages = self.construct_input_messages_with_harmony(request)?;

        let conversation = Conversation::from_messages(all_messages.clone());
        let token_ids = self
            .encoding
            .render_conversation_for_completion(&conversation, Role::Assistant, None)
            .map_err(|e| format!("Failed to encode Harmony conversation: {}", e))?;

        let selection_text = self.extract_selection_text(&all_messages);

        // Get stop tokens for Harmony assistant actions (<|return|> and <|call|>)
        let stop_token_ids: Vec<u32> = self
            .encoding
            .stop_tokens_for_assistant_actions()
            .into_iter()
            .flat_map(|set| set.into_iter())
            .collect();

        // Decode tokens to see what the model actually receives
        let decoded_text = self
            .encoding
            .tokenizer()
            .decode_utf8(&token_ids)
            .unwrap_or_else(|_| "<decode error>".to_string());
        debug!(
            token_count = token_ids.len(),
            token_preview = ?&token_ids[..token_ids.len().min(20)],
            decoded_length = decoded_text.len(),
            "Encoded conversation to tokens - decoded text follows:"
        );
        debug!("DECODED_TEXT_START\n{}\nDECODED_TEXT_END", decoded_text);

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
    /// Build system message with common logic
    ///
    /// # Arguments
    /// * `reasoning_effort` - Optional reasoning effort level
    /// * `has_tools` - Whether custom tools are present
    fn build_system_message(
        &self,
        reasoning_effort: Option<ReasoningEffort>,
        has_tools: bool,
    ) -> HarmonyMessage {
        let mut sys_content = SystemContent::new();

        // Add reasoning_effort if provided
        if let Some(effort) = reasoning_effort {
            sys_content = sys_content.with_reasoning_effort(effort);
        }

        // Set conversation start date (always current date)
        sys_content =
            sys_content.with_conversation_start_date(Local::now().format("%Y-%m-%d").to_string());

        // If no tools, remove "commentary" from valid channels
        if !has_tools {
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

    fn build_system_message_from_chat(&self, request: &ChatCompletionRequest) -> HarmonyMessage {
        let reasoning_effort = request
            .reasoning_effort
            .as_deref()
            .map(|effort| match effort {
                "high" => ReasoningEffort::High,
                "medium" => ReasoningEffort::Medium,
                "low" => ReasoningEffort::Low,
                // Harmony does not support minimal reasoning effort
                "minimal" => ReasoningEffort::Low,
                _ => ReasoningEffort::Medium,
            });

        let has_tools = request.tools.is_some();
        self.build_system_message(reasoning_effort, has_tools)
    }

    /// Build system message from ResponsesRequest
    ///
    /// # Arguments
    /// * `request` - The ResponsesRequest
    /// * `with_custom_tools` - Whether custom tools (beyond built-ins) are present
    fn build_system_message_from_responses(
        &self,
        request: &ResponsesRequest,
        with_custom_tools: bool,
    ) -> HarmonyMessage {
        let reasoning_effort = request
            .reasoning
            .as_ref()
            .and_then(|r| r.effort.as_ref())
            .map(|effort| match effort {
                ResponsesReasoningEffort::High => ReasoningEffort::High,
                ResponsesReasoningEffort::Medium => ReasoningEffort::Medium,
                ResponsesReasoningEffort::Low => ReasoningEffort::Low,
                ResponsesReasoningEffort::Minimal => ReasoningEffort::Low,
            });

        self.build_system_message(reasoning_effort, with_custom_tools)
    }

    /// Build developer message with common logic
    ///
    /// Filters out built-in tools and converts custom tools to ToolDescription
    ///
    /// # Arguments
    /// * `tools` - Optional list of tools
    /// * `instructions` - Optional instructions (Responses API only)
    fn build_developer_message<T: ToolLike>(
        &self,
        tools: Option<&Vec<T>>,
        instructions: Option<&str>,
    ) -> HarmonyMessage {
        let mut dev_content = DeveloperContent::new();

        // Add instructions if provided (Responses API only)
        if let Some(instructions) = instructions {
            dev_content = dev_content.with_instructions(instructions.to_string());
        }

        // Early return if no tools
        let Some(tools) = tools else {
            return HarmonyMessage::from_role_and_content(Role::Developer, dev_content);
        };

        // Filter to custom tools and convert to ToolDescription
        let tool_descriptions: Vec<ToolDescription> = tools
            .iter()
            .filter(|t| t.is_custom())
            .filter_map(|t| t.to_tool_description())
            .collect();

        // Add function tools to developer content
        if !tool_descriptions.is_empty() {
            dev_content = dev_content.with_function_tools(tool_descriptions);
        }

        HarmonyMessage::from_role_and_content(Role::Developer, dev_content)
    }

    fn build_developer_message_from_chat(&self, tools: Option<&Vec<Tool>>) -> HarmonyMessage {
        self.build_developer_message(tools, None)
    }

    /// Build developer message from Responses request
    ///
    /// # Arguments
    /// * `instructions` - Optional instructions (Responses API specific)
    /// * `tools` - Optional list of tools
    fn build_developer_message_from_responses(
        &self,
        instructions: Option<&str>,
        tools: Option<&Vec<ResponseTool>>,
    ) -> HarmonyMessage {
        self.build_developer_message(tools, instructions)
    }

    /// Construct input messages for Responses API with Harmony
    ///
    /// Handles both new conversations and continuations of previous responses.
    ///
    /// This handles:
    /// - New conversation: system message, developer message, and user input
    /// - Continuing conversation: loads previous messages, cleans up chain-of-thoughts
    /// - MCP tool allowlisting for special tool types
    /// - Complex response input parsing with function call tracking
    ///
    /// # Arguments
    /// * `request` - The ResponsesRequest
    /// * `prev_response` - Optional previous response to continue from
    fn construct_input_messages_with_harmony(
        &self,
        request: &ResponsesRequest,
    ) -> Result<Vec<HarmonyMessage>, String> {
        let mut all_messages = Vec::new();

        // Handle new vs continuing conversation
        if request.previous_response_id.is_none() {
            // New conversation

            let tool_types: Vec<&str> = request
                .tools
                .as_ref()
                .map(|tools| {
                    tools
                        .iter()
                        .map(|tool| match tool.r#type {
                            ResponseToolType::Function => "function",
                            ResponseToolType::WebSearchPreview => "web_search_preview",
                            ResponseToolType::CodeInterpreter => "code_interpreter",
                            ResponseToolType::Mcp => "mcp",
                        })
                        .collect()
                })
                .unwrap_or_default();

            let with_custom_tools = has_custom_tools(&tool_types);

            // Add system message
            let sys_msg = self.build_system_message_from_responses(request, with_custom_tools);
            all_messages.push(sys_msg);

            // Add developer message only if we have custom tools
            if with_custom_tools {
                let dev_msg = self.build_developer_message_from_responses(
                    request.instructions.as_deref(),
                    request.tools.as_ref(),
                );
                all_messages.push(dev_msg);
            }
        } else {
            // Continue the previous conversation
            // NOTE: Previous messages are loaded by serve_harmony_responses() before calling this method.
            // The request.input will already contain the conversation history when previous_response_id was set.
            // We just proceed with parsing the input items as normal.
            debug!("Continuing conversation (history already loaded in request.input)");
        }

        // Append the new input
        // Responses API supports simple text inputs without chat format
        match &request.input {
            ResponseInput::Text(text) => {
                let user_msg = HarmonyMessage {
                    author: Author {
                        role: Role::User,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text: text.clone() })],
                    channel: None,
                    content_type: None,
                };
                all_messages.push(user_msg);
            }
            ResponseInput::Items(items) => {
                // Track function calls for looking up call_id → name mapping
                let mut prev_outputs: Vec<&ResponseInputOutputItem> = Vec::new();

                for item in items {
                    let msg = self.parse_response_item_to_harmony_message(item, &prev_outputs)?;
                    all_messages.push(msg);

                    // Track function tool calls so that function_call_output can find the name
                    if matches!(item, ResponseInputOutputItem::FunctionToolCall { .. }) {
                        prev_outputs.push(item);
                    }
                }
            }
        }

        debug!(
            message_count = all_messages.len(),
            "Constructed Harmony messages for Responses API"
        );
        Ok(all_messages)
    }

    /// Parse a ResponseInputOutputItem into a HarmonyMessage
    ///
    /// Handles conversion of various response item types (messages, function calls, reasoning, etc.)
    /// to Harmony message format.
    ///
    /// # Arguments
    /// * `item` - The ResponseInputOutputItem to parse
    /// * `prev_outputs` - Previous items for looking up function call names (for function_call_output)
    fn parse_response_item_to_harmony_message(
        &self,
        item: &ResponseInputOutputItem,
        prev_outputs: &[&ResponseInputOutputItem],
    ) -> Result<HarmonyMessage, String> {
        match item {
            // Regular message (user or assistant)
            ResponseInputOutputItem::Message { role, content, .. } => {
                let harmony_role = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "system" => Role::System,
                    _ => Role::User, // Default to user for unknown roles
                };

                // Extract text from content parts
                let text_parts: Vec<String> = content
                    .iter()
                    .filter_map(|part| match part {
                        ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                        ResponseContentPart::InputText { text } => Some(text.clone()),
                        ResponseContentPart::Unknown => None,
                    })
                    .collect();

                let text = text_parts.join("\n");

                Ok(HarmonyMessage {
                    author: Author {
                        role: harmony_role,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text })],
                    channel: None,
                    content_type: None,
                })
            }

            // Reasoning content (chain-of-thought)
            ResponseInputOutputItem::Reasoning { content, .. } => {
                // Extract reasoning text
                let reasoning_texts: Vec<String> = content
                    .iter()
                    .map(|rc| match rc {
                        ResponseReasoningContent::ReasoningText { text } => text.clone(),
                    })
                    .collect();

                let text = reasoning_texts.join("\n");

                // Reasoning goes in the "analysis" channel for Harmony
                Ok(HarmonyMessage {
                    author: Author {
                        role: Role::Assistant,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text })],
                    channel: Some("analysis".to_string()),
                    content_type: None,
                })
            }

            // Function tool call (with optional output)
            ResponseInputOutputItem::FunctionToolCall {
                name,
                arguments,
                output,
                ..
            } => {
                // If there's an output, this represents the tool result
                // Otherwise, it's the tool call itself
                if let Some(output_str) = output {
                    // Tool result - use Tool role with "functions.{name}" as author name
                    // IMPORTANT: Must include recipient="assistant" for parser to recognize it.
                    // We keep channel=None to minimize what the model might copy.
                    let author_name = format!("functions.{}", name);
                    debug!(
                        tool_name = %name,
                        author_name = %author_name,
                        output_preview = %output_str.chars().take(100).collect::<String>(),
                        "Building tool result message with Tool role (recipient=assistant, no channel)"
                    );
                    Ok(HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(author_name),
                        },
                        recipient: Some("assistant".to_string()),
                        content: vec![Content::Text(TextContent {
                            text: output_str.clone(),
                        })],
                        channel: None,
                        content_type: None,
                    })
                } else {
                    // Tool call - assistant message in commentary channel with recipient
                    // msg.with_channel("commentary").with_recipient(f"functions.{name}")
                    let recipient = format!("functions.{}", name);
                    debug!(
                        tool_name = %name,
                        recipient = %recipient,
                        "Building tool call message with recipient"
                    );
                    Ok(HarmonyMessage {
                        author: Author {
                            role: Role::Assistant,
                            name: None,
                        },
                        recipient: Some(recipient),
                        content: vec![Content::Text(TextContent {
                            text: arguments.clone(),
                        })],
                        channel: Some("commentary".to_string()),
                        content_type: Some("json".to_string()),
                    })
                }
            }

            // Function call output (separate from call) - requires looking up the original call
            ResponseInputOutputItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                // Search prev_outputs in reverse order to find the matching function call
                let call = prev_outputs
                    .iter()
                    .rev()
                    .find_map(|item| match item {
                        ResponseInputOutputItem::FunctionToolCall { id, name, .. }
                            if id == call_id =>
                        {
                            Some(name.clone())
                        }
                        _ => None,
                    })
                    .ok_or_else(|| format!("No function call found for call_id: {}", call_id))?;

                // Create Tool message with "functions.{name}" prefix
                // IMPORTANT: Must include recipient="assistant" for parser to recognize it.
                // We keep channel=None to minimize what the model might copy.
                Ok(HarmonyMessage {
                    author: Author {
                        role: Role::Tool,
                        name: Some(format!("functions.{}", call)),
                    },
                    recipient: Some("assistant".to_string()),
                    content: vec![Content::Text(TextContent {
                        text: output.clone(),
                    })],
                    channel: None,
                    content_type: None,
                })
            }

            // Simple input message (usually user message)
            ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                let harmony_role = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "system" => Role::System,
                    _ => Role::User,
                };

                let text = match content {
                    StringOrContentParts::String(s) => s.clone(),
                    StringOrContentParts::Array(parts) => {
                        // Extract text from content parts
                        parts
                            .iter()
                            .filter_map(|part| match part {
                                ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                                ResponseContentPart::InputText { text } => Some(text.clone()),
                                ResponseContentPart::Unknown => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };

                Ok(HarmonyMessage {
                    author: Author {
                        role: harmony_role,
                        name: None,
                    },
                    recipient: None,
                    content: vec![Content::Text(TextContent { text })],
                    channel: None,
                    content_type: None,
                })
            }
        }
    }

    /// Convert OpenAI ChatMessage format to Harmony messages
    ///
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
                    // Look up the function name from the tool_call_id
                    let function_name = tool_call_map
                        .get(tool_call_id)
                        .cloned()
                        .unwrap_or_else(|| tool_call_id.clone());

                    // Tool result - Must include recipient="assistant" for parser to recognize it.
                    // We keep channel=None to minimize what the model might copy.
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(format!("functions.{}", function_name)),
                        },
                        recipient: Some("assistant".to_string()),
                        content: vec![Content::Text(TextContent {
                            text: content.clone(),
                        })],
                        channel: None,
                        content_type: None,
                    };
                    harmony_messages.push(harmony_msg);
                }

                ChatMessage::Function { content, name } => {
                    // Function messages also use Role::Tool
                    // Tool result - Must include recipient="assistant" for parser to recognize it.
                    // We keep channel=None to minimize what the model might copy.
                    let harmony_msg = HarmonyMessage {
                        author: Author {
                            role: Role::Tool,
                            name: Some(format!("functions.{}", name)),
                        },
                        recipient: Some("assistant".to_string()),
                        content: vec![Content::Text(TextContent {
                            text: content.clone(),
                        })],
                        channel: None,
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
    /// Uses the last user message for load balancing
    fn extract_selection_text(&self, messages: &[HarmonyMessage]) -> String {
        // Find the last user message
        if let Some(last_user_msg) = messages.iter().rev().find(|m| m.author.role == Role::User) {
            // Extract full text from content
            return last_user_msg
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::Text(tc) => Some(tc.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
        }

        // Fallback: concatenate all text
        messages
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::Text(tc) => Some(tc.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for HarmonyBuilder {
    fn default() -> Self {
        Self::new()
    }
}
