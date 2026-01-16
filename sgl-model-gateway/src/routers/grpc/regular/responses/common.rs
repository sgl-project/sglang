//! Shared helpers and state tracking for Regular Responses
//!
//! This module contains common utilities used by both streaming and non-streaming paths:
//! - ToolLoopState for tracking multi-turn tool calling
//! - Helper functions for tool preparation and extraction
//! - MCP metadata builders
//! - Conversation history loading

use std::{collections::HashMap, sync::Arc};

use axum::response::Response;
use serde_json::{json, Value};
use tracing::{debug, warn};
use uuid::Uuid;

pub(super) use crate::routers::mcp_utils::{
    build_server_label_map, decode_mcp_function_name, encode_mcp_function_name,
    filter_tools_for_server, list_tools_by_server, resolve_server_label, McpToolLookup,
};
use crate::{
    data_connector::{self, ConversationId, ResponseId},
    mcp::{self, McpManager},
    protocols::{
        chat::ChatCompletionRequest,
        common::{Function, Tool, ToolChoice, ToolChoiceValue},
        responses::{
            self, McpToolInfo, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponseOutputItem, ResponseTool, ResponsesRequest,
        },
    },
    routers::{
        error, grpc::common::responses::ResponsesContext, mcp_utils::build_allowed_tools_map,
    },
};

// ============================================================================
// Tool Loop State
// ============================================================================

/// State for tracking multi-turn tool calling loop
pub(super) struct ToolLoopState {
    pub iteration: usize,
    pub total_calls: usize,
    pub conversation_history: Vec<ResponseInputOutputItem>,
    pub original_input: ResponseInput,
    pub mcp_call_items: Vec<ResponseOutputItem>,
    pub tool_lookup: McpToolLookup,
}

impl ToolLoopState {
    pub fn new(original_input: ResponseInput, tool_lookup: McpToolLookup) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
            mcp_call_items: Vec::new(),
            tool_lookup,
        }
    }

    pub fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        success: bool,
        error: Option<String>,
    ) {
        // Add function_tool_call item with both arguments and output
        self.conversation_history
            .push(ResponseInputOutputItem::FunctionToolCall {
                id: call_id.clone(),
                call_id: call_id.clone(),
                name: tool_name.clone(),
                arguments: args_json_str.clone(),
                output: Some(output_str.clone()),
                status: Some("completed".to_string()),
            });

        let (resolved_label, raw_tool_name) = decode_mcp_function_name(&tool_name)
            .unwrap_or_else(|| ("mcp".to_string(), tool_name.clone()));
        let mcp_call = build_mcp_call_item(
            &raw_tool_name,
            &args_json_str,
            &output_str,
            &resolved_label,
            success,
            error.as_deref(),
        );
        self.mcp_call_items.push(mcp_call);
    }
}

// ============================================================================
// Tool Preparation and Extraction
// ============================================================================

/// Merge function tools from request with MCP tools and set tool_choice based on iteration
pub(super) fn prepare_chat_tools_and_choice(
    chat_request: &mut ChatCompletionRequest,
    mcp_chat_tools: &[Tool],
    iteration: usize,
) {
    // Merge function tools from request with MCP tools
    let mut all_tools = chat_request.tools.clone().unwrap_or_default();
    all_tools.extend(mcp_chat_tools.iter().cloned());
    chat_request.tools = Some(all_tools);

    // Set tool_choice based on iteration
    // - Iteration 0: Use user's tool_choice or default to auto
    // - Iteration 1+: Always use auto to avoid infinite loops
    chat_request.tool_choice = if iteration == 0 {
        chat_request
            .tool_choice
            .clone()
            .or(Some(ToolChoice::Value(ToolChoiceValue::Auto)))
    } else {
        Some(ToolChoice::Value(ToolChoiceValue::Auto))
    };
}

/// Tool call extracted from a ChatCompletionResponse
#[derive(Debug, Clone)]
pub(super) struct ExtractedToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

/// Extract all tool calls from chat response (for parallel tool call support)
pub(super) fn extract_all_tool_calls_from_chat(
    response: &crate::protocols::chat::ChatCompletionResponse,
) -> Vec<ExtractedToolCall> {
    // Check if response has choices with tool calls
    let Some(choice) = response.choices.first() else {
        return Vec::new();
    };
    let message = &choice.message;

    // Look for tool_calls in the message
    if let Some(tool_calls) = &message.tool_calls {
        tool_calls
            .iter()
            .map(|tool_call| ExtractedToolCall {
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call
                    .function
                    .arguments
                    .clone()
                    .unwrap_or_else(|| "{}".to_string()),
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// Convert MCP tools to Chat API tool format and build lookup
pub(super) fn convert_mcp_tools_to_chat_tools(
    mcp: &Arc<McpManager>,
    server_keys: &[String],
    server_labels: &HashMap<String, String>,
    request_tools: Option<&[ResponseTool]>,
) -> (Vec<Tool>, McpToolLookup) {
    let mut tools = Vec::new();
    let mut lookup = McpToolLookup::default();
    let allowed_tools = build_allowed_tools_map(request_tools);

    for (server_key, tools_for_server) in list_tools_by_server(mcp, server_keys) {
        let server_label = resolve_server_label(&server_key, server_labels);
        let filtered_tools =
            filter_tools_for_server(&tools_for_server, &server_label, &allowed_tools);

        if filtered_tools.is_empty() {
            continue;
        }

        for tool_info in filtered_tools {
            let encoded_name = encode_mcp_function_name(&server_label, tool_info.name.as_ref());
            let parameters = Value::Object((*tool_info.input_schema).clone());

            tools.push(Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: encoded_name.clone(),
                    description: tool_info.description.as_ref().map(|d| d.to_string()),
                    parameters: parameters.clone(),
                    strict: None,
                },
            });

            lookup
                .tool_servers
                .insert(encoded_name.clone(), server_key.clone());
            lookup
                .tool_names
                .insert(encoded_name.clone(), tool_info.name.to_string());
            lookup.tool_schemas.insert(encoded_name, parameters);
        }
    }

    (tools, lookup)
}

// ============================================================================
// MCP Metadata Builders
// ============================================================================

/// Generate unique ID for MCP items
pub(super) fn generate_mcp_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

fn build_mcp_list_tools_item_from_tools(
    server_label: &str,
    tools: &[mcp::Tool],
) -> ResponseOutputItem {
    let tools_info: Vec<McpToolInfo> = tools
        .iter()
        .map(|t| McpToolInfo {
            name: t.name.to_string(),
            description: t.description.as_ref().map(|d| d.to_string()),
            input_schema: Value::Object((*t.input_schema).clone()),
            annotations: Some(json!({
                "read_only": false
            })),
        })
        .collect();

    ResponseOutputItem::McpListTools {
        id: generate_mcp_id("mcpl"),
        server_label: server_label.to_string(),
        tools: tools_info,
    }
}

pub(super) fn build_mcp_list_tools_items_for_request(
    mcp: &Arc<McpManager>,
    server_keys: &[String],
    tools: Option<&[ResponseTool]>,
) -> Vec<ResponseOutputItem> {
    let server_labels = build_server_label_map(tools);
    let allowed_tools = build_allowed_tools_map(tools);

    list_tools_by_server(mcp, server_keys)
        .into_iter()
        .map(|(server_key, tools)| {
            let server_label = resolve_server_label(&server_key, &server_labels);
            let filtered = filter_tools_for_server(&tools, &server_label, &allowed_tools);
            build_mcp_list_tools_item_from_tools(&server_label, &filtered)
        })
        .collect()
}

/// Build mcp_call output item
pub(super) fn build_mcp_call_item(
    tool_name: &str,
    arguments: &str,
    output: &str,
    server_label: &str,
    success: bool,
    error: Option<&str>,
) -> ResponseOutputItem {
    ResponseOutputItem::McpCall {
        id: generate_mcp_id("mcp"),
        status: if success { "completed" } else { "failed" }.to_string(),
        approval_request_id: None,
        arguments: arguments.to_string(),
        error: error.map(|e| e.to_string()),
        name: tool_name.to_string(),
        output: output.to_string(),
        server_label: server_label.to_string(),
    }
}

// ============================================================================
// Conversation History Loading
// ============================================================================

/// Load conversation history and response chains, returning modified request
pub(super) async fn load_conversation_history(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
) -> Result<ResponsesRequest, Response> {
    let mut modified_request = request.clone();
    let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;

    // Handle previous_response_id by loading response chain
    if let Some(ref prev_id_str) = modified_request.previous_response_id {
        let prev_id = ResponseId::from(prev_id_str.as_str());
        match ctx
            .response_storage
            .get_response_chain(&prev_id, None)
            .await
        {
            Ok(chain) => {
                let mut items = Vec::new();
                for stored in chain.responses.iter() {
                    // Convert input items from stored input (which is now a JSON array)
                    if let Some(input_arr) = stored.input.as_array() {
                        for item in input_arr {
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
                                Ok(input_item) => {
                                    items.push(input_item);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to deserialize stored input item: {}. Item: {}",
                                        e, item
                                    );
                                }
                            }
                        }
                    }

                    // Convert output items from stored output (which is now a JSON array)
                    if let Some(output_arr) = stored.output.as_array() {
                        for item in output_arr {
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
                                Ok(output_item) => {
                                    items.push(output_item);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to deserialize stored output item: {}. Item: {}",
                                        e, item
                                    );
                                }
                            }
                        }
                    }
                }
                conversation_items = Some(items);
                modified_request.previous_response_id = None;
            }
            Err(e) => {
                warn!(
                    "Failed to load previous response chain for {}: {}",
                    prev_id_str, e
                );
            }
        }
    }

    // Handle conversation by loading conversation history
    if let Some(ref conv_id_str) = request.conversation {
        let conv_id = ConversationId::from(conv_id_str.as_str());

        // Check if conversation exists - return error if not found
        let conversation = ctx
            .conversation_storage
            .get_conversation(&conv_id)
            .await
            .map_err(|e| {
                error::internal_error(
                    "check_conversation_failed",
                    format!("Failed to check conversation: {}", e),
                )
            })?;

        if conversation.is_none() {
            return Err(error::not_found(
                "conversation_not_found",
                format!(
                    "Conversation '{}' not found. Please create the conversation first using the conversations API.",
                    conv_id_str
                )
            ));
        }

        // Load conversation history
        const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;
        let params = data_connector::ListParams {
            limit: MAX_CONVERSATION_HISTORY_ITEMS,
            order: data_connector::SortOrder::Asc,
            after: None,
        };

        match ctx
            .conversation_item_storage
            .list_items(&conv_id, params)
            .await
        {
            Ok(stored_items) => {
                let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                for item in stored_items.into_iter() {
                    if item.item_type == "message" {
                        if let Ok(content_parts) =
                            serde_json::from_value::<Vec<ResponseContentPart>>(item.content.clone())
                        {
                            items.push(ResponseInputOutputItem::Message {
                                id: item.id.0.clone(),
                                role: item.role.clone().unwrap_or_else(|| "user".to_string()),
                                content: content_parts,
                                status: item.status.clone(),
                            });
                        }
                    }
                }

                // Append current request
                match &modified_request.input {
                    ResponseInput::Text(text) => {
                        items.push(ResponseInputOutputItem::Message {
                            id: format!("msg_u_{}", conv_id.0),
                            role: "user".to_string(),
                            content: vec![ResponseContentPart::InputText { text: text.clone() }],
                            status: Some("completed".to_string()),
                        });
                    }
                    ResponseInput::Items(current_items) => {
                        // Process all item types, converting SimpleInputMessage to Message
                        for item in current_items.iter() {
                            let normalized = responses::normalize_input_item(item);
                            items.push(normalized);
                        }
                    }
                }

                modified_request.input = ResponseInput::Items(items);
            }
            Err(e) => {
                warn!("Failed to load conversation history: {}", e);
            }
        }
    }

    // If we have conversation_items from previous_response_id, merge them
    if let Some(mut items) = conversation_items {
        // Append current request
        match &modified_request.input {
            ResponseInput::Text(text) => {
                items.push(ResponseInputOutputItem::Message {
                    id: format!(
                        "msg_u_{}",
                        request
                            .previous_response_id
                            .as_ref()
                            .unwrap_or(&"new".to_string())
                    ),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                });
            }
            ResponseInput::Items(current_items) => {
                // Process all item types, converting SimpleInputMessage to Message
                for item in current_items.iter() {
                    let normalized = responses::normalize_input_item(item);
                    items.push(normalized);
                }
            }
        }

        modified_request.input = ResponseInput::Items(items);
    }

    debug!(
        has_previous_response = request.previous_response_id.is_some(),
        has_conversation = request.conversation.is_some(),
        "Loaded conversation history"
    );

    Ok(modified_request)
}

/// Build next request with updated conversation history
pub(super) fn build_next_request(
    state: &ToolLoopState,
    current_request: &ResponsesRequest,
) -> ResponsesRequest {
    // Start with original input
    let mut input_items = match &state.original_input {
        ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
            id: format!("msg_u_{}", state.iteration),
            role: "user".to_string(),
            content: vec![ResponseContentPart::InputText { text: text.clone() }],
            status: Some("completed".to_string()),
        }],
        ResponseInput::Items(items) => items.iter().map(responses::normalize_input_item).collect(),
    };

    // Append all conversation history (function calls and outputs)
    input_items.extend_from_slice(&state.conversation_history);

    // Build new request for next iteration
    ResponsesRequest {
        input: ResponseInput::Items(input_items),
        model: current_request.model.clone(),
        instructions: current_request.instructions.clone(),
        tools: current_request.tools.clone(),
        max_output_tokens: current_request.max_output_tokens,
        temperature: current_request.temperature,
        top_p: current_request.top_p,
        stream: current_request.stream,
        store: Some(false), // Don't store intermediate responses
        background: Some(false),
        max_tool_calls: current_request.max_tool_calls,
        tool_choice: current_request.tool_choice.clone(),
        parallel_tool_calls: current_request.parallel_tool_calls,
        previous_response_id: None,
        conversation: None,
        user: current_request.user.clone(),
        metadata: current_request.metadata.clone(),
        include: current_request.include.clone(),
        reasoning: current_request.reasoning.clone(),
        service_tier: current_request.service_tier.clone(),
        top_logprobs: current_request.top_logprobs,
        truncation: current_request.truncation.clone(),
        text: current_request.text.clone(),
        request_id: None,
        priority: current_request.priority,
        frequency_penalty: current_request.frequency_penalty,
        presence_penalty: current_request.presence_penalty,
        stop: current_request.stop.clone(),
        top_k: current_request.top_k,
        min_p: current_request.min_p,
        repetition_penalty: current_request.repetition_penalty,
    }
}
