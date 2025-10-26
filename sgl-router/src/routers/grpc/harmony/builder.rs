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
    responses::{
        ReasoningEffort as ResponsesReasoningEffort, ResponseInput, ResponseToolType,
        ResponsesRequest,
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

/// Check if there are custom tools beyond built-in ones
///
/// Following vLLM's has_custom_tools() logic:
/// ```python
/// def has_custom_tools(tool_types: list[str]) -> bool:
///     return not set(tool_types).issubset(BUILTIN_TOOLS)
/// ```
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

    /// Build system message from ResponsesRequest
    ///
    /// Follows vLLM's get_system_message() logic for Responses API
    ///
    /// # Arguments
    /// * `request` - The ResponsesRequest
    /// * `with_custom_tools` - Whether custom tools (beyond built-ins) are present
    fn build_system_message_from_responses(
        &self,
        request: &ResponsesRequest,
        with_custom_tools: bool,
    ) -> HarmonyMessage {
        let mut sys_content = SystemContent::new();

        // Add instructions (Responses API)
        if let Some(instructions) = request.instructions.as_deref() {
            sys_content = sys_content.with_model_identity(instructions.to_string());
        }

        // Extract reasoning effort from Responses API
        if let Some(reasoning) = &request.reasoning {
            if let Some(effort) = &reasoning.effort {
                let effort_enum = match effort {
                    ResponsesReasoningEffort::High => ReasoningEffort::High,
                    ResponsesReasoningEffort::Medium => ReasoningEffort::Medium,
                    ResponsesReasoningEffort::Low => ReasoningEffort::Low,
                };
                sys_content = sys_content.with_reasoning_effort(effort_enum);
            }
        }

        // Set conversation start date (current date)
        sys_content =
            sys_content.with_conversation_start_date(Local::now().format("%Y-%m-%d").to_string());

        // If no custom tools, remove "commentary" from valid channels
        // Following vLLM's logic
        if !with_custom_tools {
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

    /// Get developer message for Responses API
    ///
    /// Follows vLLM's get_developer_message() logic for Responses API
    ///
    /// # Arguments
    /// * `instructions` - Optional instructions (Responses API specific, handled in system message)
    /// * `tools` - Optional list of tools
    fn get_developer_message_from_responses(
        &self,
        _instructions: Option<&str>,
        tools: Option<&Vec<crate::protocols::responses::ResponseTool>>,
    ) -> HarmonyMessage {
        let dev_content = DeveloperContent::new();

        // Note: Instructions are handled in build_system_message_from_responses,
        // added to model_identity, following vLLM's get_system_message() logic

        // Early return if no tools
        let Some(tools) = tools else {
            return HarmonyMessage::from_role_and_content(Role::Developer, dev_content);
        };

        // Filter tools - skip built-in tools, only add function tools
        // Following vLLM's logic:
        // - web_search_preview, code_interpreter, container, mcp are built-in (skip)
        // - function tools are added to developer message
        // Note: Currently Responses API doesn't have function tools,
        // TODO: Implement function tool descriptions when available

        for tool in tools {
            match tool.r#type {
                ResponseToolType::WebSearchPreview
                | ResponseToolType::CodeInterpreter
                | ResponseToolType::Mcp => {
                    // These are built-in tools that are added to the system message.
                    // Skip them in developer message.
                }
            }
        }

        HarmonyMessage::from_role_and_content(Role::Developer, dev_content)
    }

    /// Construct input messages for Responses API with Harmony
    ///
    /// Follows vLLM's _construct_input_messages_with_harmony() logic for Responses API.
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
                let dev_msg = self.get_developer_message_from_responses(
                    request.instructions.as_deref(),
                    request.tools.as_ref(),
                );
                all_messages.push(dev_msg);
            }
        } else {
            // Continue the previous conversation
            // NOTE: Currently, request params like reasoning and instructions are ignored
            // TODO: Load previous messages from storage (msg_store) when available

            // For now, this is a placeholder - full implementation requires:
            // 1. Access to msg_store from previous response
            // 2. Chain-of-thoughts cleanup logic
            // 3. Proper state management across turns
            tracing::debug!(
                "Continuing conversation from previous response: {:?}",
                request.previous_response_id
            );
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
                // Following vLLM's parse_response_input() logic for converting items
                // This needs to maintain prev_outputs state for function call tracking
                let prev_outputs = Vec::new();

                for item in items {
                    let msg = self.parse_response_item_to_harmony_message(item, &prev_outputs)?;
                    all_messages.push(msg);

                    // Track function calls for output lookup
                    // TODO: Properly deserialize and track ResponseFunctionToolCall items
                }
            }
        }

        Ok(all_messages)
    }

    /// Parse a ResponseInputOutputItem into a HarmonyMessage
    ///
    /// Follows vLLM's parse_response_input() logic.
    /// Handles conversion of various response item types (messages, function calls, reasoning, etc.)
    /// to Harmony message format.
    ///
    /// # Arguments
    /// * `item` - The ResponseInputOutputItem to parse
    /// * `prev_outputs` - Previous outputs for tracking function call context
    fn parse_response_item_to_harmony_message(
        &self,
        item: &crate::protocols::responses::ResponseInputOutputItem,
        _prev_outputs: &[crate::protocols::responses::ResponseOutputItem],
    ) -> Result<HarmonyMessage, String> {
        // TODO: Full implementation needed to match vLLM's parse_response_input() logic
        // This should handle:
        // - Regular messages (user/assistant/system)
        // - Function call outputs with call_id lookup
        // - Reasoning items
        // - Function calls with arguments
        // - Proper channel and recipient assignment

        // For now, handle basic message types
        // Default: treat as text content user message
        let text_content = serde_json::to_string(item)
            .map_err(|e| format!("Failed to serialize response item: {}", e))?;

        let msg = HarmonyMessage {
            author: Author {
                role: Role::User,
                name: None,
            },
            recipient: None,
            content: vec![Content::Text(TextContent { text: text_content })],
            channel: None,
            content_type: None,
        };

        Ok(msg)
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
