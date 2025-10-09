//! OpenAI router implementation

use crate::config::CircuitBreakerConfig;
use crate::core::{CircuitBreaker, CircuitBreakerConfig as CoreCircuitBreakerConfig};
use crate::data_connector::{
    Conversation, ConversationId, ConversationItemsListParams, ConversationItemsSortOrder,
    ConversationMetadata, NewConversationItem as DCNewConversationItem, ResponseId,
    SharedConversationItemStorage, SharedConversationStorage, SharedResponseStorage,
    StoredResponse,
};
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, RerankRequest,
    ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
    ResponseStatus, ResponseTextFormat, ResponseTool, ResponseToolType, ResponsesGetParams,
    ResponsesRequest, ResponsesResponse, TextFormatType,
};
use crate::routers::header_utils::{apply_request_headers, preserve_response_headers};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, to_value, Value};
use std::{
    any::Any,
    borrow::Cow,
    collections::HashMap,
    io,
    sync::{atomic::AtomicBool, Arc},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info, warn};

// SSE Event Type Constants - single source of truth for event type strings
mod event_types {
    // Response lifecycle events
    pub const RESPONSE_CREATED: &str = "response.created";
    pub const RESPONSE_IN_PROGRESS: &str = "response.in_progress";
    pub const RESPONSE_COMPLETED: &str = "response.completed";

    // Output item events
    pub const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
    pub const OUTPUT_ITEM_DONE: &str = "response.output_item.done";
    pub const OUTPUT_ITEM_DELTA: &str = "response.output_item.delta";

    // Function call events
    pub const FUNCTION_CALL_ARGUMENTS_DELTA: &str = "response.function_call_arguments.delta";
    pub const FUNCTION_CALL_ARGUMENTS_DONE: &str = "response.function_call_arguments.done";

    // MCP call events
    pub const MCP_CALL_ARGUMENTS_DELTA: &str = "response.mcp_call_arguments.delta";
    pub const MCP_CALL_ARGUMENTS_DONE: &str = "response.mcp_call_arguments.done";
    pub const MCP_CALL_IN_PROGRESS: &str = "response.mcp_call.in_progress";
    pub const MCP_CALL_COMPLETED: &str = "response.mcp_call.completed";
    pub const MCP_LIST_TOOLS_IN_PROGRESS: &str = "response.mcp_list_tools.in_progress";
    pub const MCP_LIST_TOOLS_COMPLETED: &str = "response.mcp_list_tools.completed";

    // Item types
    pub const ITEM_TYPE_FUNCTION_CALL: &str = "function_call";
    pub const ITEM_TYPE_FUNCTION_TOOL_CALL: &str = "function_tool_call";
    pub const ITEM_TYPE_MCP_CALL: &str = "mcp_call";
    pub const ITEM_TYPE_FUNCTION: &str = "function";
    pub const ITEM_TYPE_MCP_LIST_TOOLS: &str = "mcp_list_tools";
}

/// Router for OpenAI backend
pub struct OpenAIRouter {
    /// HTTP client for upstream OpenAI-compatible API
    client: reqwest::Client,
    /// Base URL for identification (no trailing slash)
    base_url: String,
    /// Circuit breaker
    circuit_breaker: CircuitBreaker,
    /// Health status
    healthy: AtomicBool,
    /// Response storage for managing conversation history
    response_storage: SharedResponseStorage,
    /// Conversation storage backend
    conversation_storage: SharedConversationStorage,
    /// Conversation item storage backend
    conversation_item_storage: SharedConversationItemStorage,
    /// Optional MCP manager (enabled via config presence)
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
}

impl std::fmt::Debug for OpenAIRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIRouter")
            .field("base_url", &self.base_url)
            .field("healthy", &self.healthy)
            .finish()
    }
}

/// Configuration for MCP tool calling loops
#[derive(Debug, Clone)]
struct McpLoopConfig {
    /// Maximum iterations as safety limit (internal only, default: 10)
    /// Prevents infinite loops when max_tool_calls is not set
    max_iterations: usize,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self { max_iterations: 10 }
    }
}

/// State for tracking multi-turn tool calling loop
struct ToolLoopState {
    /// Current iteration number (starts at 0, increments with each tool call)
    iteration: usize,
    /// Total number of tool calls executed
    total_calls: usize,
    /// Conversation history (function_call and function_call_output items)
    conversation_history: Vec<Value>,
    /// Original user input (preserved for building resume payloads)
    original_input: ResponseInput,
}

impl ToolLoopState {
    fn new(original_input: ResponseInput) -> Self {
        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
        }
    }

    /// Record a tool call in the loop state
    fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
    ) {
        // Add function_call item to history
        let func_item = json!({
            "type": event_types::ITEM_TYPE_FUNCTION_CALL,
            "call_id": call_id,
            "name": tool_name,
            "arguments": args_json_str
        });
        self.conversation_history.push(func_item);

        // Add function_call_output item to history
        let output_item = json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_str
        });
        self.conversation_history.push(output_item);
    }
}

/// Helper that parses SSE frames from the OpenAI responses stream and
/// accumulates enough information to persist the final response locally.
struct StreamingResponseAccumulator {
    /// The initial `response.created` payload (if emitted).
    initial_response: Option<Value>,
    /// The final `response.completed` payload (if emitted).
    completed_response: Option<Value>,
    /// Collected output items keyed by the upstream output index, used when
    /// a final response payload is absent and we need to synthesize one.
    output_items: Vec<(usize, Value)>,
    /// Captured error payload (if the upstream stream fails midway).
    encountered_error: Option<Value>,
}

/// Represents a function call being accumulated across delta events
#[derive(Debug, Clone)]
struct FunctionCallInProgress {
    call_id: String,
    name: String,
    arguments_buffer: String,
    output_index: usize,
    last_obfuscation: Option<String>,
    assigned_output_index: Option<usize>,
}

impl FunctionCallInProgress {
    fn new(call_id: String, output_index: usize) -> Self {
        Self {
            call_id,
            name: String::new(),
            arguments_buffer: String::new(),
            output_index,
            last_obfuscation: None,
            assigned_output_index: None,
        }
    }

    fn is_complete(&self) -> bool {
        // A tool call is complete if it has a name
        !self.name.is_empty()
    }

    fn effective_output_index(&self) -> usize {
        self.assigned_output_index.unwrap_or(self.output_index)
    }
}

#[derive(Debug, Default)]
struct OutputIndexMapper {
    next_index: usize,
    // Map upstream output_index -> remapped output_index
    assigned: HashMap<usize, usize>,
}

impl OutputIndexMapper {
    fn with_start(next_index: usize) -> Self {
        Self {
            next_index,
            assigned: HashMap::new(),
        }
    }

    fn ensure_mapping(&mut self, upstream_index: usize) -> usize {
        *self.assigned.entry(upstream_index).or_insert_with(|| {
            let assigned = self.next_index;
            self.next_index += 1;
            assigned
        })
    }

    fn lookup(&self, upstream_index: usize) -> Option<usize> {
        self.assigned.get(&upstream_index).copied()
    }

    fn allocate_synthetic(&mut self) -> usize {
        let assigned = self.next_index;
        self.next_index += 1;
        assigned
    }

    fn next_index(&self) -> usize {
        self.next_index
    }
}

/// Action to take based on streaming event processing
#[derive(Debug)]
enum StreamAction {
    Forward,      // Pass event to client
    Buffer,       // Accumulate for tool execution
    ExecuteTools, // Function call complete, execute now
}

/// Handles streaming responses with MCP tool call interception
struct StreamingToolHandler {
    /// Accumulator for response persistence
    accumulator: StreamingResponseAccumulator,
    /// Function calls being built from deltas
    pending_calls: Vec<FunctionCallInProgress>,
    /// Track if we're currently in a function call
    in_function_call: bool,
    /// Manage output_index remapping so they increment per item
    output_index_mapper: OutputIndexMapper,
    /// Original response id captured from the first response.created event
    original_response_id: Option<String>,
}

impl StreamingToolHandler {
    fn with_starting_index(start: usize) -> Self {
        Self {
            accumulator: StreamingResponseAccumulator::new(),
            pending_calls: Vec::new(),
            in_function_call: false,
            output_index_mapper: OutputIndexMapper::with_start(start),
            original_response_id: None,
        }
    }

    fn ensure_output_index(&mut self, upstream_index: usize) -> usize {
        self.output_index_mapper.ensure_mapping(upstream_index)
    }

    fn mapped_output_index(&self, upstream_index: usize) -> Option<usize> {
        self.output_index_mapper.lookup(upstream_index)
    }

    fn allocate_synthetic_output_index(&mut self) -> usize {
        self.output_index_mapper.allocate_synthetic()
    }

    fn next_output_index(&self) -> usize {
        self.output_index_mapper.next_index()
    }

    fn original_response_id(&self) -> Option<&str> {
        self.original_response_id
            .as_deref()
            .or_else(|| self.accumulator.original_response_id())
    }

    fn snapshot_final_response(&self) -> Option<Value> {
        self.accumulator.snapshot_final_response()
    }

    /// Process an SSE event and determine what action to take
    fn process_event(&mut self, event_name: Option<&str>, data: &str) -> StreamAction {
        // Always feed to accumulator for storage
        self.accumulator.ingest_block(&format!(
            "{}data: {}",
            event_name
                .map(|n| format!("event: {}\n", n))
                .unwrap_or_default(),
            data
        ));

        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return StreamAction::Forward,
        };

        let event_type = event_name
            .map(|s| s.to_string())
            .or_else(|| {
                parsed
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();

        match event_type.as_str() {
            event_types::RESPONSE_CREATED => {
                if self.original_response_id.is_none() {
                    if let Some(response_obj) = parsed.get("response").and_then(|v| v.as_object()) {
                        if let Some(id) = response_obj.get("id").and_then(|v| v.as_str()) {
                            self.original_response_id = Some(id.to_string());
                        }
                    }
                }
                StreamAction::Forward
            }
            event_types::RESPONSE_COMPLETED => StreamAction::Forward,
            event_types::OUTPUT_ITEM_ADDED => {
                if let Some(idx) = parsed.get("output_index").and_then(|v| v.as_u64()) {
                    self.ensure_output_index(idx as usize);
                }

                // Check if this is a function_call item being added
                if let Some(item) = parsed.get("item") {
                    if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                        if item_type == event_types::ITEM_TYPE_FUNCTION_CALL
                            || item_type == event_types::ITEM_TYPE_FUNCTION_TOOL_CALL
                        {
                            match parsed.get("output_index").and_then(|v| v.as_u64()) {
                                Some(idx) => {
                                    let output_index = idx as usize;
                                    let assigned_index = self.ensure_output_index(output_index);
                                    let call_id =
                                        item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                                    let name =
                                        item.get("name").and_then(|v| v.as_str()).unwrap_or("");

                                    // Create or update the function call
                                    let call = self.get_or_create_call(output_index, item);
                                    call.call_id = call_id.to_string();
                                    call.name = name.to_string();
                                    call.assigned_output_index = Some(assigned_index);

                                    self.in_function_call = true;
                                }
                                None => {
                                    tracing::warn!(
                                        "Missing output_index in function_call added event, \
                                         forwarding without processing for tool execution"
                                    );
                                }
                            }
                        }
                    }
                }
                StreamAction::Forward
            }
            event_types::FUNCTION_CALL_ARGUMENTS_DELTA => {
                // Accumulate arguments for the function call
                if let Some(output_index) = parsed
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                {
                    let assigned_index = self.ensure_output_index(output_index);
                    if let Some(delta) = parsed.get("delta").and_then(|v| v.as_str()) {
                        if let Some(call) = self
                            .pending_calls
                            .iter_mut()
                            .find(|c| c.output_index == output_index)
                        {
                            call.arguments_buffer.push_str(delta);
                            if let Some(obfuscation) =
                                parsed.get("obfuscation").and_then(|v| v.as_str())
                            {
                                call.last_obfuscation = Some(obfuscation.to_string());
                            }
                            if call.assigned_output_index.is_none() {
                                call.assigned_output_index = Some(assigned_index);
                            }
                        }
                    }
                }
                StreamAction::Forward
            }
            event_types::FUNCTION_CALL_ARGUMENTS_DONE => {
                // Function call arguments complete - check if ready to execute
                if let Some(output_index) = parsed
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                {
                    let assigned_index = self.ensure_output_index(output_index);
                    if let Some(call) = self
                        .pending_calls
                        .iter_mut()
                        .find(|c| c.output_index == output_index)
                    {
                        if call.assigned_output_index.is_none() {
                            call.assigned_output_index = Some(assigned_index);
                        }
                    }
                }

                if self.has_complete_calls() {
                    StreamAction::ExecuteTools
                } else {
                    StreamAction::Forward
                }
            }
            event_types::OUTPUT_ITEM_DELTA => self.process_output_delta(&parsed),
            event_types::OUTPUT_ITEM_DONE => {
                // Check if we have complete function calls ready to execute
                if let Some(output_index) = parsed
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                {
                    self.ensure_output_index(output_index);
                }

                if self.has_complete_calls() {
                    StreamAction::ExecuteTools
                } else {
                    StreamAction::Forward
                }
            }
            _ => StreamAction::Forward,
        }
    }

    /// Process output delta events to detect and accumulate function calls
    fn process_output_delta(&mut self, event: &Value) -> StreamAction {
        let output_index = event
            .get("output_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let assigned_index = self.ensure_output_index(output_index);

        let delta = match event.get("delta") {
            Some(d) => d,
            None => return StreamAction::Forward,
        };

        // Check if this is a function call delta
        let item_type = delta.get("type").and_then(|v| v.as_str());

        if item_type == Some(event_types::ITEM_TYPE_FUNCTION_TOOL_CALL)
            || item_type == Some(event_types::ITEM_TYPE_FUNCTION_CALL)
        {
            self.in_function_call = true;

            // Get or create function call for this output index
            let call = self.get_or_create_call(output_index, delta);
            call.assigned_output_index = Some(assigned_index);

            // Accumulate call_id if present
            if let Some(call_id) = delta.get("call_id").and_then(|v| v.as_str()) {
                call.call_id = call_id.to_string();
            }

            // Accumulate name if present
            if let Some(name) = delta.get("name").and_then(|v| v.as_str()) {
                call.name.push_str(name);
            }

            // Accumulate arguments if present
            if let Some(args) = delta.get("arguments").and_then(|v| v.as_str()) {
                call.arguments_buffer.push_str(args);
            }

            if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
                call.last_obfuscation = Some(obfuscation.to_string());
            }

            // Buffer this event, don't forward to client
            return StreamAction::Buffer;
        }

        // Forward non-function-call events
        StreamAction::Forward
    }

    fn get_or_create_call(
        &mut self,
        output_index: usize,
        delta: &Value,
    ) -> &mut FunctionCallInProgress {
        // Find existing call for this output index
        // Note: We use position() + index instead of iter_mut().find() because we need
        // to potentially mutate pending_calls after the early return, which causes
        // borrow checker issues with the iter_mut approach
        if let Some(pos) = self
            .pending_calls
            .iter()
            .position(|c| c.output_index == output_index)
        {
            return &mut self.pending_calls[pos];
        }

        // Create new call
        let call_id = delta
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut call = FunctionCallInProgress::new(call_id, output_index);
        if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
            call.last_obfuscation = Some(obfuscation.to_string());
        }

        self.pending_calls.push(call);
        self.pending_calls
            .last_mut()
            .expect("Just pushed to pending_calls, must have at least one element")
    }

    fn has_complete_calls(&self) -> bool {
        !self.pending_calls.is_empty() && self.pending_calls.iter().all(|c| c.is_complete())
    }

    fn take_pending_calls(&mut self) -> Vec<FunctionCallInProgress> {
        std::mem::take(&mut self.pending_calls)
    }
}

impl StreamingResponseAccumulator {
    fn new() -> Self {
        Self {
            initial_response: None,
            completed_response: None,
            output_items: Vec::new(),
            encountered_error: None,
        }
    }

    /// Feed the accumulator with the next SSE chunk.
    fn ingest_block(&mut self, block: &str) {
        if block.trim().is_empty() {
            return;
        }
        self.process_block(block);
    }

    /// Consume the accumulator and produce the best-effort final response value.
    fn into_final_response(mut self) -> Option<Value> {
        if self.completed_response.is_some() {
            return self.completed_response;
        }

        self.build_fallback_response()
    }

    fn encountered_error(&self) -> Option<&Value> {
        self.encountered_error.as_ref()
    }

    fn original_response_id(&self) -> Option<&str> {
        self.initial_response
            .as_ref()
            .and_then(|response| response.get("id"))
            .and_then(|id| id.as_str())
    }

    fn snapshot_final_response(&self) -> Option<Value> {
        if let Some(resp) = &self.completed_response {
            return Some(resp.clone());
        }
        self.build_fallback_response_snapshot()
    }

    fn build_fallback_response_snapshot(&self) -> Option<Value> {
        let mut response = self.initial_response.clone()?;

        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));

            let mut output_items = self.output_items.clone();
            output_items.sort_by_key(|(index, _)| *index);
            let outputs: Vec<Value> = output_items.into_iter().map(|(_, item)| item).collect();
            obj.insert("output".to_string(), Value::Array(outputs));
        }

        Some(response)
    }

    fn process_block(&mut self, block: &str) {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            return;
        }

        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        for line in trimmed.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start().to_string());
            }
        }

        let data_payload = data_lines.join("\n");
        if data_payload.is_empty() {
            return;
        }

        self.handle_event(event_name.as_deref(), &data_payload);
    }

    fn handle_event(&mut self, event_name: Option<&str>, data_payload: &str) {
        let parsed: Value = match serde_json::from_str(data_payload) {
            Ok(value) => value,
            Err(err) => {
                warn!("Failed to parse streaming event JSON: {}", err);
                return;
            }
        };

        let event_type = event_name
            .map(|s| s.to_string())
            .or_else(|| {
                parsed
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();

        match event_type.as_str() {
            event_types::RESPONSE_CREATED => {
                if self.initial_response.is_none() {
                    if let Some(response) = parsed.get("response") {
                        self.initial_response = Some(response.clone());
                    }
                }
            }
            event_types::RESPONSE_COMPLETED => {
                if let Some(response) = parsed.get("response") {
                    self.completed_response = Some(response.clone());
                }
            }
            event_types::OUTPUT_ITEM_DONE => {
                if let (Some(index), Some(item)) = (
                    parsed
                        .get("output_index")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    parsed.get("item"),
                ) {
                    self.output_items.push((index, item.clone()));
                }
            }
            "response.error" => {
                self.encountered_error = Some(parsed);
            }
            _ => {}
        }
    }

    fn build_fallback_response(&mut self) -> Option<Value> {
        let mut response = self.initial_response.clone()?;

        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));

            self.output_items.sort_by_key(|(index, _)| *index);
            let outputs: Vec<Value> = self
                .output_items
                .iter()
                .map(|(_, item)| item.clone())
                .collect();
            obj.insert("output".to_string(), Value::Array(outputs));
        }

        Some(response)
    }
}

impl OpenAIRouter {
    // Maximum number of conversation items to attach as input when a conversation is provided
    const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;
    /// Create a new OpenAI router
    pub async fn new(
        base_url: String,
        circuit_breaker_config: Option<CircuitBreakerConfig>,
        response_storage: SharedResponseStorage,
        conversation_storage: SharedConversationStorage,
        conversation_item_storage: SharedConversationItemStorage,
    ) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let base_url = base_url.trim_end_matches('/').to_string();

        // Convert circuit breaker config
        let core_cb_config = circuit_breaker_config
            .map(|cb| CoreCircuitBreakerConfig {
                failure_threshold: cb.failure_threshold,
                success_threshold: cb.success_threshold,
                timeout_duration: std::time::Duration::from_secs(cb.timeout_duration_secs),
                window_duration: std::time::Duration::from_secs(cb.window_duration_secs),
            })
            .unwrap_or_default();

        let circuit_breaker = CircuitBreaker::with_config(core_cb_config);

        // Optional MCP manager activation via env var path (config-driven gate)
        let mcp_manager = match std::env::var("SGLANG_MCP_CONFIG").ok() {
            Some(path) if !path.trim().is_empty() => {
                match crate::mcp::McpConfig::from_file(&path).await {
                    Ok(cfg) => match crate::mcp::McpClientManager::new(cfg).await {
                        Ok(mgr) => Some(Arc::new(mgr)),
                        Err(err) => {
                            warn!("Failed to initialize MCP manager: {}", err);
                            None
                        }
                    },
                    Err(err) => {
                        warn!("Failed to load MCP config from '{}': {}", path, err);
                        None
                    }
                }
            }
            _ => None,
        };

        Ok(Self {
            client,
            base_url,
            circuit_breaker,
            healthy: AtomicBool::new(true),
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
        })
    }

    async fn handle_non_streaming_response(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        mut payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        // Request-scoped MCP: build from request tools if provided; otherwise fall back to router-level MCP
        let req_mcp_manager = Self::mcp_manager_from_request_tools(&original_body.tools).await;
        let active_mcp = req_mcp_manager.as_ref().or(self.mcp_manager.as_ref());

        // If the client requested MCP but we couldn't initialize it, fail early with a clear error
        let requested_mcp = original_body
            .tools
            .iter()
            .any(|t| matches!(t.r#type, ResponseToolType::Mcp));
        if requested_mcp && active_mcp.is_none() {
            return (
                StatusCode::BAD_GATEWAY,
                json!({
                    "error": {
                        "message": "MCP server unavailable or failed to initialize from request tools",
                        "type": "mcp_unavailable",
                        "param": "tools",
                    }
                })
                .to_string(),
            )
                .into_response();
        }

        // If MCP is active, mirror one function tool into the outgoing payload
        if let Some(mcp) = active_mcp {
            if let Some(obj) = payload.as_object_mut() {
                // Remove any non-function tools (e.g., custom "mcp" items) from outgoing payload
                if let Some(v) = obj.get_mut("tools") {
                    if let Some(arr) = v.as_array_mut() {
                        arr.retain(|item| {
                            item.get("type")
                                .and_then(|v| v.as_str())
                                .map(|s| s == "function")
                                .unwrap_or(false)
                        });
                        if arr.is_empty() {
                            obj.remove("tools");
                            obj.insert(
                                "tool_choice".to_string(),
                                Value::String("none".to_string()),
                            );
                        }
                    }
                }
                // Build function tools for all discovered MCP tools
                let mut tools_json = Vec::new();
                let tools = mcp.list_tools();
                for t in tools {
                    let parameters = t.parameters.clone().unwrap_or(serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }));
                    let tool = serde_json::json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": parameters
                    });
                    tools_json.push(tool);
                }
                if !tools_json.is_empty() {
                    obj.insert("tools".to_string(), Value::Array(tools_json));
                    // Ensure tool_choice auto to allow model planning
                    obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
                }
            }
        }
        let request_builder = self.client.post(&url).json(&payload);

        // Apply headers with filtering
        let request_builder = if let Some(headers) = headers {
            apply_request_headers(headers, request_builder, true)
        } else {
            request_builder
        };

        match request_builder.send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|e| format!("Failed to get error body: {}", e));
                    return (status, error_text).into_response();
                }

                // Parse the response
                match response.json::<Value>().await {
                    Ok(mut openai_response_json) => {
                        if let Some(prev_id) = original_previous_response_id {
                            if let Some(obj) = openai_response_json.as_object_mut() {
                                let should_insert = obj
                                    .get("previous_response_id")
                                    .map(|v| v.is_null())
                                    .unwrap_or(true);
                                if should_insert {
                                    obj.insert(
                                        "previous_response_id".to_string(),
                                        Value::String(prev_id),
                                    );
                                }
                            }
                        }

                        if let Some(obj) = openai_response_json.as_object_mut() {
                            if !obj.contains_key("instructions") {
                                if let Some(instructions) = &original_body.instructions {
                                    obj.insert(
                                        "instructions".to_string(),
                                        Value::String(instructions.clone()),
                                    );
                                }
                            }

                            if !obj.contains_key("metadata") {
                                if let Some(metadata) = &original_body.metadata {
                                    let metadata_map: serde_json::Map<String, Value> = metadata
                                        .iter()
                                        .map(|(k, v)| (k.clone(), v.clone()))
                                        .collect();
                                    obj.insert("metadata".to_string(), Value::Object(metadata_map));
                                }
                            }

                            // Reflect the client's requested store preference in the response body
                            obj.insert("store".to_string(), Value::Bool(original_body.store));
                        }

                        // If MCP is active and we detect a function call, enter the tool loop
                        let mut final_response_json = if let Some(mcp) = active_mcp {
                            if Self::extract_function_call(&openai_response_json).is_some() {
                                // Use the loop to handle potentially multiple tool calls
                                let loop_config = McpLoopConfig::default();
                                match self
                                    .execute_tool_loop(
                                        &url,
                                        headers,
                                        payload.clone(),
                                        original_body,
                                        mcp,
                                        &loop_config,
                                    )
                                    .await
                                {
                                    Ok(loop_result) => loop_result,
                                    Err(err) => {
                                        warn!("Tool loop failed: {}", err);
                                        let error_body = json!({
                                            "error": {
                                                "message": format!("Tool loop failed: {}", err),
                                                "type": "internal_error",
                                            }
                                        })
                                        .to_string();
                                        return (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            [("content-type", "application/json")],
                                            error_body,
                                        )
                                            .into_response();
                                    }
                                }
                            } else {
                                // No function call detected, use response as-is
                                openai_response_json
                            }
                        } else {
                            openai_response_json
                        };

                        // Mask tools back to MCP format for client
                        Self::mask_tools_as_mcp(&mut final_response_json, original_body);
                        // Attach conversation id for client response if present (not forwarded upstream)
                        if let Some(conv_id) = original_body.conversation.clone() {
                            if let Some(obj) = final_response_json.as_object_mut() {
                                obj.insert("conversation".to_string(), json!({"id": conv_id}));
                            }
                        }
                        if original_body.store {
                            if let Err(e) = self
                                .store_response_internal(&final_response_json, original_body)
                                .await
                            {
                                warn!("Failed to store response: {}", e);
                            }
                        }
                        if let Some(conv_id) = original_body.conversation.clone() {
                            if let Err(err) = self
                                .persist_conversation_items(
                                    &conv_id,
                                    original_body,
                                    &final_response_json,
                                )
                                .await
                            {
                                warn!("Failed to persist conversation items: {}", err);
                            }
                        }

                        match serde_json::to_string(&final_response_json) {
                            Ok(json_str) => (
                                StatusCode::OK,
                                [("content-type", "application/json")],
                                json_str,
                            )
                                .into_response(),
                            Err(e) => {
                                error!("Failed to serialize response: {}", e);
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    json!({"error": {"message": "Failed to serialize response", "type": "internal_error"}}).to_string(),
                                )
                                    .into_response()
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to parse OpenAI response: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to parse response: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Failed to forward request to OpenAI: {}", e),
            )
                .into_response(),
        }
    }

    async fn persist_conversation_items(
        &self,
        conversation_id: &str,
        original_body: &ResponsesRequest,
        final_response_json: &Value,
    ) -> Result<(), String> {
        persist_items_with_storages(
            self.conversation_storage.clone(),
            self.conversation_item_storage.clone(),
            conversation_id.to_string(),
            original_body.clone(),
            final_response_json.clone(),
        )
        .await
    }

    /// Build a request-scoped MCP manager from request tools, if present.
    async fn mcp_manager_from_request_tools(
        tools: &[ResponseTool],
    ) -> Option<Arc<crate::mcp::McpClientManager>> {
        let tool = tools
            .iter()
            .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())?;
        let server_url = tool.server_url.as_ref()?.trim().to_string();
        if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
            warn!(
                "Ignoring MCP server_url with unsupported scheme: {}",
                server_url
            );
            return None;
        }
        let name = tool
            .server_label
            .clone()
            .unwrap_or_else(|| "request-mcp".to_string());
        let token = tool.authorization.clone();
        let transport = if server_url.contains("/sse") {
            crate::mcp::McpTransport::Sse {
                url: server_url,
                token,
            }
        } else {
            crate::mcp::McpTransport::Streamable {
                url: server_url,
                token,
            }
        };
        let cfg = crate::mcp::McpConfig {
            servers: vec![crate::mcp::McpServerConfig { name, transport }],
        };
        match crate::mcp::McpClientManager::new(cfg).await {
            Ok(mgr) => Some(Arc::new(mgr)),
            Err(err) => {
                warn!("Failed to initialize request-scoped MCP manager: {}", err);
                None
            }
        }
    }

    async fn handle_streaming_response(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        // Check if MCP is active for this request
        let req_mcp_manager = Self::mcp_manager_from_request_tools(&original_body.tools).await;
        let active_mcp = req_mcp_manager.as_ref().or(self.mcp_manager.as_ref());

        // If no MCP is active, use simple pass-through streaming
        if active_mcp.is_none() {
            return self
                .handle_simple_streaming_passthrough(
                    url,
                    headers,
                    payload,
                    original_body,
                    original_previous_response_id,
                )
                .await;
        }

        let active_mcp = active_mcp.unwrap();

        // MCP is active - transform tools and set up interception
        self.handle_streaming_with_tool_interception(
            url,
            headers,
            payload,
            original_body,
            original_previous_response_id,
            active_mcp,
        )
        .await
    }

    /// Simple pass-through streaming without MCP interception
    async fn handle_simple_streaming_passthrough(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
    ) -> Response {
        let mut request_builder = self.client.post(&url).json(&payload);

        if let Some(headers) = headers {
            request_builder = apply_request_headers(headers, request_builder, true);
        }

        request_builder = request_builder.header("Accept", "text/event-stream");

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.circuit_breaker.record_failure();
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to forward request to OpenAI: {}", err),
                )
                    .into_response();
            }
        };

        let status = response.status();
        let status_code =
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !status.is_success() {
            self.circuit_breaker.record_failure();
            let error_body = match response.text().await {
                Ok(body) => body,
                Err(err) => format!("Failed to read upstream error body: {}", err),
            };
            return (status_code, error_body).into_response();
        }

        self.circuit_breaker.record_success();

        let preserved_headers = preserve_response_headers(response.headers());
        let mut upstream_stream = response.bytes_stream();

        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        let should_store = original_body.store;
        let storage = self.response_storage.clone();
        let conv_storage = self.conversation_storage.clone();
        let conv_item_storage = self.conversation_item_storage.clone();
        let original_request = original_body.clone();
        let persist_needed = original_request.conversation.is_some();
        let previous_response_id = original_previous_response_id.clone();

        tokio::spawn(async move {
            let mut accumulator = StreamingResponseAccumulator::new();
            let mut upstream_failed = false;
            let mut receiver_connected = true;
            let mut pending = String::new();

            while let Some(chunk_result) = upstream_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_text = match std::str::from_utf8(&chunk) {
                            Ok(text) => Cow::Borrowed(text),
                            Err(_) => Cow::Owned(String::from_utf8_lossy(&chunk).to_string()),
                        };

                        pending.push_str(&chunk_text.replace("\r\n", "\n"));

                        while let Some(pos) = pending.find("\n\n") {
                            let raw_block = pending[..pos].to_string();
                            pending.drain(..pos + 2);

                            if raw_block.trim().is_empty() {
                                continue;
                            }

                            let block_cow = if let Some(modified) = Self::rewrite_streaming_block(
                                raw_block.as_str(),
                                &original_request,
                                previous_response_id.as_deref(),
                            ) {
                                Cow::Owned(modified)
                            } else {
                                Cow::Borrowed(raw_block.as_str())
                            };

                            if should_store || persist_needed {
                                accumulator.ingest_block(block_cow.as_ref());
                            }

                            if receiver_connected {
                                let chunk_to_send = format!("{}\n\n", block_cow);
                                if tx.send(Ok(Bytes::from(chunk_to_send))).is_err() {
                                    receiver_connected = false;
                                }
                            }

                            if !receiver_connected && !should_store {
                                break;
                            }
                        }

                        if !receiver_connected && !should_store {
                            break;
                        }
                    }
                    Err(err) => {
                        upstream_failed = true;
                        let io_err = io::Error::other(err);
                        let _ = tx.send(Err(io_err));
                        break;
                    }
                }
            }

            if (should_store || persist_needed) && !upstream_failed {
                if !pending.trim().is_empty() {
                    accumulator.ingest_block(&pending);
                }
                let encountered_error = accumulator.encountered_error().cloned();
                if let Some(mut response_json) = accumulator.into_final_response() {
                    Self::patch_streaming_response_json(
                        &mut response_json,
                        &original_request,
                        previous_response_id.as_deref(),
                    );

                    if should_store {
                        if let Err(err) =
                            Self::store_response_impl(&storage, &response_json, &original_request)
                                .await
                        {
                            warn!("Failed to store streaming response: {}", err);
                        }
                    }
                    if persist_needed {
                        if let Some(conv_id) = original_request.conversation.clone() {
                            if let Err(err) = persist_items_with_storages(
                                conv_storage.clone(),
                                conv_item_storage.clone(),
                                conv_id,
                                original_request.clone(),
                                response_json.clone(),
                            )
                            .await
                            {
                                warn!("Failed to persist conversation items (stream): {}", err);
                            }
                        }
                    }
                } else if let Some(error_payload) = encountered_error {
                    warn!("Upstream streaming error payload: {}", error_payload);
                } else {
                    warn!("Streaming completed without a final response payload");
                }
            }
        });

        let body_stream = UnboundedReceiverStream::new(rx);
        let mut response = Response::new(Body::from_stream(body_stream));
        *response.status_mut() = status_code;

        let headers_mut = response.headers_mut();
        for (name, value) in preserved_headers.iter() {
            headers_mut.insert(name, value.clone());
        }

        if !headers_mut.contains_key(CONTENT_TYPE) {
            headers_mut.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        }

        response
    }

    /// Apply all transformations to event data in-place (rewrite + transform)
    /// Optimized to parse JSON only once instead of multiple times
    /// Returns true if any changes were made
    fn apply_event_transformations_inplace(
        parsed_data: &mut Value,
        server_label: &str,
        original_request: &ResponsesRequest,
        previous_response_id: Option<&str>,
    ) -> bool {
        let mut changed = false;

        // 1. Apply rewrite_streaming_block logic (store, previous_response_id, tools masking)
        let event_type = parsed_data
            .get("type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();

        let should_patch = matches!(
            event_type.as_str(),
            event_types::RESPONSE_CREATED
                | event_types::RESPONSE_IN_PROGRESS
                | event_types::RESPONSE_COMPLETED
        );

        if should_patch {
            if let Some(response_obj) = parsed_data
                .get_mut("response")
                .and_then(|v| v.as_object_mut())
            {
                let desired_store = Value::Bool(original_request.store);
                if response_obj.get("store") != Some(&desired_store) {
                    response_obj.insert("store".to_string(), desired_store);
                    changed = true;
                }

                if let Some(prev_id) = previous_response_id {
                    let needs_previous = response_obj
                        .get("previous_response_id")
                        .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                        .unwrap_or(true);

                    if needs_previous {
                        response_obj.insert(
                            "previous_response_id".to_string(),
                            Value::String(prev_id.to_string()),
                        );
                        changed = true;
                    }
                }

                // Mask tools from function to MCP format (optimized without cloning)
                if response_obj.get("tools").is_some() {
                    let requested_mcp = original_request
                        .tools
                        .iter()
                        .any(|t| matches!(t.r#type, ResponseToolType::Mcp));

                    if requested_mcp {
                        if let Some(mcp_tools) = Self::build_mcp_tools_value(original_request) {
                            response_obj.insert("tools".to_string(), mcp_tools);
                            response_obj
                                .entry("tool_choice".to_string())
                                .or_insert(Value::String("auto".to_string()));
                            changed = true;
                        }
                    }
                }
            }
        }

        // 2. Apply transform_streaming_event logic (function_call → mcp_call)
        match event_type.as_str() {
            event_types::OUTPUT_ITEM_ADDED | event_types::OUTPUT_ITEM_DONE => {
                if let Some(item) = parsed_data.get_mut("item") {
                    if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                        if item_type == event_types::ITEM_TYPE_FUNCTION_CALL
                            || item_type == event_types::ITEM_TYPE_FUNCTION_TOOL_CALL
                        {
                            item["type"] = json!(event_types::ITEM_TYPE_MCP_CALL);
                            item["server_label"] = json!(server_label);

                            // Transform ID from fc_* to mcp_*
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                if let Some(stripped) = id.strip_prefix("fc_") {
                                    let new_id = format!("mcp_{}", stripped);
                                    item["id"] = json!(new_id);
                                }
                            }

                            changed = true;
                        }
                    }
                }
            }
            event_types::FUNCTION_CALL_ARGUMENTS_DONE => {
                parsed_data["type"] = json!(event_types::MCP_CALL_ARGUMENTS_DONE);

                // Transform item_id from fc_* to mcp_*
                if let Some(item_id) = parsed_data.get("item_id").and_then(|v| v.as_str()) {
                    if let Some(stripped) = item_id.strip_prefix("fc_") {
                        let new_id = format!("mcp_{}", stripped);
                        parsed_data["item_id"] = json!(new_id);
                    }
                }

                changed = true;
            }
            _ => {}
        }

        changed
    }

    /// Forward and transform a streaming event to the client
    /// Returns false if client disconnected
    #[allow(clippy::too_many_arguments)]
    fn forward_streaming_event(
        raw_block: &str,
        event_name: Option<&str>,
        data: &str,
        handler: &mut StreamingToolHandler,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        server_label: &str,
        original_request: &ResponsesRequest,
        previous_response_id: Option<&str>,
        sequence_number: &mut u64,
    ) -> bool {
        // Skip individual function_call_arguments.delta events - we'll send them as one
        if event_name == Some(event_types::FUNCTION_CALL_ARGUMENTS_DELTA) {
            return true;
        }

        // Parse JSON data once (optimized!)
        let mut parsed_data: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => {
                // If parsing fails, forward raw block as-is
                let chunk_to_send = format!("{}\n\n", raw_block);
                return tx.send(Ok(Bytes::from(chunk_to_send))).is_ok();
            }
        };

        let event_type = event_name
            .or_else(|| parsed_data.get("type").and_then(|v| v.as_str()))
            .unwrap_or("");

        if event_type == event_types::RESPONSE_COMPLETED {
            return true;
        }

        // Check if this is function_call_arguments.done - need to send buffered args first
        let mut mapped_output_index: Option<usize> = None;

        if event_name == Some(event_types::FUNCTION_CALL_ARGUMENTS_DONE) {
            if let Some(output_index) = parsed_data
                .get("output_index")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
            {
                let assigned_index = handler
                    .mapped_output_index(output_index)
                    .unwrap_or(output_index);
                mapped_output_index = Some(assigned_index);

                if let Some(call) = handler
                    .pending_calls
                    .iter()
                    .find(|c| c.output_index == output_index)
                {
                    let arguments_value = if call.arguments_buffer.is_empty() {
                        "{}".to_string()
                    } else {
                        call.arguments_buffer.clone()
                    };

                    // Make sure the done event carries full arguments
                    parsed_data["arguments"] = Value::String(arguments_value.clone());

                    // Get item_id and transform it
                    let item_id = parsed_data
                        .get("item_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let mcp_item_id = if let Some(stripped) = item_id.strip_prefix("fc_") {
                        format!("mcp_{}", stripped)
                    } else {
                        item_id.to_string()
                    };

                    // Emit a synthetic MCP arguments delta event before the done event
                    let mut delta_event = json!({
                        "type": event_types::MCP_CALL_ARGUMENTS_DELTA,
                        "sequence_number": *sequence_number,
                        "output_index": assigned_index,
                        "item_id": mcp_item_id,
                        "delta": arguments_value,
                    });

                    if let Some(obfuscation) = call.last_obfuscation.as_ref() {
                        if let Some(obj) = delta_event.as_object_mut() {
                            obj.insert(
                                "obfuscation".to_string(),
                                Value::String(obfuscation.clone()),
                            );
                        }
                    } else if let Some(obfuscation) = parsed_data.get("obfuscation").cloned() {
                        if let Some(obj) = delta_event.as_object_mut() {
                            obj.insert("obfuscation".to_string(), obfuscation);
                        }
                    }

                    let delta_block = format!(
                        "event: {}\ndata: {}\n\n",
                        event_types::MCP_CALL_ARGUMENTS_DELTA,
                        delta_event
                    );
                    if tx.send(Ok(Bytes::from(delta_block))).is_err() {
                        return false;
                    }

                    *sequence_number += 1;
                }
            }
        }

        // Remap output_index (if present) so downstream sees sequential indices
        if mapped_output_index.is_none() {
            if let Some(output_index) = parsed_data
                .get("output_index")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
            {
                mapped_output_index = handler.mapped_output_index(output_index);
            }
        }

        if let Some(mapped) = mapped_output_index {
            parsed_data["output_index"] = json!(mapped);
        }

        // Apply all transformations in-place (single parse/serialize!)
        Self::apply_event_transformations_inplace(
            &mut parsed_data,
            server_label,
            original_request,
            previous_response_id,
        );

        if let Some(response_obj) = parsed_data
            .get_mut("response")
            .and_then(|v| v.as_object_mut())
        {
            if let Some(original_id) = handler.original_response_id() {
                response_obj.insert("id".to_string(), Value::String(original_id.to_string()));
            }
        }

        // Update sequence number if present in the event
        if parsed_data.get("sequence_number").is_some() {
            parsed_data["sequence_number"] = json!(*sequence_number);
            *sequence_number += 1;
        }

        // Serialize once
        let final_data = match serde_json::to_string(&parsed_data) {
            Ok(s) => s,
            Err(_) => {
                // Serialization failed, forward original
                let chunk_to_send = format!("{}\n\n", raw_block);
                return tx.send(Ok(Bytes::from(chunk_to_send))).is_ok();
            }
        };

        // Rebuild SSE block with potentially transformed event name
        let mut final_block = String::new();
        if let Some(evt) = event_name {
            // Update event name for function_call_arguments events
            if evt == event_types::FUNCTION_CALL_ARGUMENTS_DELTA {
                final_block.push_str(&format!(
                    "event: {}\n",
                    event_types::MCP_CALL_ARGUMENTS_DELTA
                ));
            } else if evt == event_types::FUNCTION_CALL_ARGUMENTS_DONE {
                final_block.push_str(&format!(
                    "event: {}\n",
                    event_types::MCP_CALL_ARGUMENTS_DONE
                ));
            } else {
                final_block.push_str(&format!("event: {}\n", evt));
            }
        }
        final_block.push_str(&format!("data: {}", final_data));

        let chunk_to_send = format!("{}\n\n", final_block);
        if tx.send(Ok(Bytes::from(chunk_to_send))).is_err() {
            return false;
        }

        // After sending output_item.added for mcp_call, inject mcp_call.in_progress event
        if event_name == Some(event_types::OUTPUT_ITEM_ADDED) {
            if let Some(item) = parsed_data.get("item") {
                if item.get("type").and_then(|v| v.as_str())
                    == Some(event_types::ITEM_TYPE_MCP_CALL)
                {
                    // Already transformed to mcp_call
                    if let (Some(item_id), Some(output_index)) = (
                        item.get("id").and_then(|v| v.as_str()),
                        parsed_data.get("output_index").and_then(|v| v.as_u64()),
                    ) {
                        let in_progress_event = json!({
                            "type": event_types::MCP_CALL_IN_PROGRESS,
                            "sequence_number": *sequence_number,
                            "output_index": output_index,
                            "item_id": item_id
                        });
                        *sequence_number += 1;
                        let in_progress_block = format!(
                            "event: {}\ndata: {}\n\n",
                            event_types::MCP_CALL_IN_PROGRESS,
                            in_progress_event
                        );
                        if tx.send(Ok(Bytes::from(in_progress_block))).is_err() {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Execute detected tool calls and send completion events to client
    /// Returns false if client disconnected during execution
    async fn execute_streaming_tool_calls(
        pending_calls: Vec<FunctionCallInProgress>,
        active_mcp: &Arc<crate::mcp::McpClientManager>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        state: &mut ToolLoopState,
        server_label: &str,
        sequence_number: &mut u64,
    ) -> bool {
        // Execute all pending tool calls (sequential, as PR3 is skipped)
        for call in pending_calls {
            // Skip if name is empty (invalid call)
            if call.name.is_empty() {
                warn!(
                    "Skipping incomplete tool call: name is empty, args_len={}",
                    call.arguments_buffer.len()
                );
                continue;
            }

            info!(
                "Executing tool call during streaming: {} ({})",
                call.name, call.call_id
            );

            // Use empty JSON object if arguments_buffer is empty
            let args_str = if call.arguments_buffer.is_empty() {
                "{}"
            } else {
                &call.arguments_buffer
            };

            let call_result = Self::execute_mcp_call(active_mcp, &call.name, args_str).await;
            let (output_str, success, error_msg) = match call_result {
                Ok((_, output)) => (output, true, None),
                Err(err) => {
                    warn!("Tool execution failed during streaming: {}", err);
                    (json!({ "error": &err }).to_string(), false, Some(err))
                }
            };

            // Send mcp_call completion event to client
            if !OpenAIRouter::send_mcp_call_completion_events_with_error(
                tx,
                &call,
                &output_str,
                server_label,
                success,
                error_msg.as_deref(),
                sequence_number,
            ) {
                // Client disconnected, no point continuing tool execution
                return false;
            }

            // Record the call
            state.record_call(call.call_id, call.name, call.arguments_buffer, output_str);
        }
        true
    }

    /// Transform payload to replace MCP tools with function tools for streaming
    fn prepare_mcp_payload_for_streaming(
        payload: &mut Value,
        active_mcp: &Arc<crate::mcp::McpClientManager>,
    ) {
        if let Some(obj) = payload.as_object_mut() {
            // Remove any non-function tools from outgoing payload
            if let Some(v) = obj.get_mut("tools") {
                if let Some(arr) = v.as_array_mut() {
                    arr.retain(|item| {
                        item.get("type")
                            .and_then(|v| v.as_str())
                            .map(|s| s == event_types::ITEM_TYPE_FUNCTION)
                            .unwrap_or(false)
                    });
                }
            }

            // Build function tools for all discovered MCP tools
            let mut tools_json = Vec::new();
            let tools = active_mcp.list_tools();
            for t in tools {
                let parameters = t.parameters.clone().unwrap_or(serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }));
                let tool = serde_json::json!({
                    "type": event_types::ITEM_TYPE_FUNCTION,
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters
                });
                tools_json.push(tool);
            }
            if !tools_json.is_empty() {
                obj.insert("tools".to_string(), Value::Array(tools_json));
                obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
            }
        }
    }

    /// Handle streaming WITH MCP tool call interception and execution
    async fn handle_streaming_with_tool_interception(
        &self,
        url: String,
        headers: Option<&HeaderMap>,
        mut payload: Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<String>,
        active_mcp: &Arc<crate::mcp::McpClientManager>,
    ) -> Response {
        // Transform MCP tools to function tools in payload
        Self::prepare_mcp_payload_for_streaming(&mut payload, active_mcp);

        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
        let should_store = original_body.store;
        let storage = self.response_storage.clone();
        let conv_storage = self.conversation_storage.clone();
        let conv_item_storage = self.conversation_item_storage.clone();
        let original_request = original_body.clone();
        let persist_needed = original_request.conversation.is_some();
        let previous_response_id = original_previous_response_id.clone();

        let client = self.client.clone();
        let url_clone = url.clone();
        let headers_opt = headers.cloned();
        let payload_clone = payload.clone();
        let active_mcp_clone = Arc::clone(active_mcp);

        // Spawn the streaming loop task
        tokio::spawn(async move {
            let mut state = ToolLoopState::new(original_request.input.clone());
            let loop_config = McpLoopConfig::default();
            let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);
            let tools_json = payload_clone.get("tools").cloned().unwrap_or(json!([]));
            let base_payload = payload_clone.clone();
            let mut current_payload = payload_clone;
            let mut mcp_list_tools_sent = false;
            let mut is_first_iteration = true;
            let mut sequence_number: u64 = 0; // Track global sequence number across all iterations
            let mut next_output_index: usize = 0;
            let mut preserved_response_id: Option<String> = None;

            let server_label = original_request
                .tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.as_deref())
                .unwrap_or("mcp");

            loop {
                // Make streaming request
                let mut request_builder = client.post(&url_clone).json(&current_payload);
                if let Some(ref h) = headers_opt {
                    request_builder = apply_request_headers(h, request_builder, true);
                }
                request_builder = request_builder.header("Accept", "text/event-stream");

                let response = match request_builder.send().await {
                    Ok(r) => r,
                    Err(e) => {
                        let error_event = format!(
                            "event: error\ndata: {{\"error\": {{\"message\": \"{}\"}}}}\n\n",
                            e
                        );
                        let _ = tx.send(Ok(Bytes::from(error_event)));
                        return;
                    }
                };

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    let error_event = format!("event: error\ndata: {{\"error\": {{\"message\": \"Upstream error {}: {}\"}}}}\n\n", status, body);
                    let _ = tx.send(Ok(Bytes::from(error_event)));
                    return;
                }

                // Stream events and check for tool calls
                let mut upstream_stream = response.bytes_stream();
                let mut handler = StreamingToolHandler::with_starting_index(next_output_index);
                if let Some(ref id) = preserved_response_id {
                    handler.original_response_id = Some(id.clone());
                }
                let mut pending = String::new();
                let mut tool_calls_detected = false;
                let mut seen_in_progress = false;

                while let Some(chunk_result) = upstream_stream.next().await {
                    match chunk_result {
                        Ok(chunk) => {
                            let chunk_text = match std::str::from_utf8(&chunk) {
                                Ok(text) => Cow::Borrowed(text),
                                Err(_) => Cow::Owned(String::from_utf8_lossy(&chunk).to_string()),
                            };

                            pending.push_str(&chunk_text.replace("\r\n", "\n"));

                            while let Some(pos) = pending.find("\n\n") {
                                let raw_block = pending[..pos].to_string();
                                pending.drain(..pos + 2);

                                if raw_block.trim().is_empty() {
                                    continue;
                                }

                                // Parse event
                                let (event_name, data) = Self::parse_sse_block(&raw_block);

                                if data.is_empty() {
                                    continue;
                                }

                                // Process through handler
                                let action = handler.process_event(event_name, data.as_ref());

                                match action {
                                    StreamAction::Forward => {
                                        // Skip response.created and response.in_progress on subsequent iterations
                                        // Do NOT consume their sequence numbers - we want continuous numbering
                                        let should_skip = if !is_first_iteration {
                                            if let Ok(parsed) =
                                                serde_json::from_str::<Value>(data.as_ref())
                                            {
                                                matches!(
                                                    parsed.get("type").and_then(|v| v.as_str()),
                                                    Some(event_types::RESPONSE_CREATED)
                                                        | Some(event_types::RESPONSE_IN_PROGRESS)
                                                )
                                            } else {
                                                false
                                            }
                                        } else {
                                            false
                                        };

                                        if !should_skip {
                                            // Forward the event
                                            if !Self::forward_streaming_event(
                                                &raw_block,
                                                event_name,
                                                data.as_ref(),
                                                &mut handler,
                                                &tx,
                                                server_label,
                                                &original_request,
                                                previous_response_id.as_deref(),
                                                &mut sequence_number,
                                            ) {
                                                // Client disconnected
                                                return;
                                            }
                                        }

                                        // After forwarding response.in_progress, send mcp_list_tools events (once)
                                        if !seen_in_progress {
                                            if let Ok(parsed) =
                                                serde_json::from_str::<Value>(data.as_ref())
                                            {
                                                if parsed.get("type").and_then(|v| v.as_str())
                                                    == Some(event_types::RESPONSE_IN_PROGRESS)
                                                {
                                                    seen_in_progress = true;
                                                    if !mcp_list_tools_sent {
                                                        let list_tools_index = handler
                                                            .allocate_synthetic_output_index();
                                                        if !OpenAIRouter::send_mcp_list_tools_events(
                                                            &tx,
                                                            &active_mcp_clone,
                                                            server_label,
                                                            list_tools_index,
                                                            &mut sequence_number,
                                                        ) {
                                                            // Client disconnected
                                                            return;
                                                        }
                                                        mcp_list_tools_sent = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    StreamAction::Buffer => {
                                        // Don't forward, just buffer
                                    }
                                    StreamAction::ExecuteTools => {
                                        if !Self::forward_streaming_event(
                                            &raw_block,
                                            event_name,
                                            data.as_ref(),
                                            &mut handler,
                                            &tx,
                                            server_label,
                                            &original_request,
                                            previous_response_id.as_deref(),
                                            &mut sequence_number,
                                        ) {
                                            // Client disconnected
                                            return;
                                        }
                                        tool_calls_detected = true;
                                        break; // Exit stream processing to execute tools
                                    }
                                }
                            }

                            if tool_calls_detected {
                                break;
                            }
                        }
                        Err(e) => {
                            let error_event = format!("event: error\ndata: {{\"error\": {{\"message\": \"Stream error: {}\"}}}}\n\n", e);
                            let _ = tx.send(Ok(Bytes::from(error_event)));
                            return;
                        }
                    }
                }

                next_output_index = handler.next_output_index();
                if let Some(id) = handler.original_response_id().map(|s| s.to_string()) {
                    preserved_response_id = Some(id);
                }

                // If no tool calls, we're done - stream is complete
                if !tool_calls_detected {
                    if !Self::send_final_response_event(
                        &handler,
                        &tx,
                        &mut sequence_number,
                        &state,
                        Some(&active_mcp_clone),
                        &original_request,
                        previous_response_id.as_deref(),
                        server_label,
                    ) {
                        return;
                    }

                    let final_response_json = if should_store || persist_needed {
                        handler.accumulator.into_final_response()
                    } else {
                        None
                    };

                    if let Some(mut response_json) = final_response_json {
                        if let Some(ref id) = preserved_response_id {
                            if let Some(obj) = response_json.as_object_mut() {
                                obj.insert("id".to_string(), Value::String(id.clone()));
                            }
                        }
                        Self::inject_mcp_metadata_streaming(
                            &mut response_json,
                            &state,
                            &active_mcp_clone,
                            server_label,
                        );

                        Self::mask_tools_as_mcp(&mut response_json, &original_request);
                        Self::patch_streaming_response_json(
                            &mut response_json,
                            &original_request,
                            previous_response_id.as_deref(),
                        );

                        if should_store {
                            if let Err(err) = Self::store_response_impl(
                                &storage,
                                &response_json,
                                &original_request,
                            )
                            .await
                            {
                                warn!("Failed to store streaming response: {}", err);
                            }
                        }

                        if persist_needed {
                            if let Some(conv_id) = original_request.conversation.clone() {
                                if let Err(err) = persist_items_with_storages(
                                    conv_storage.clone(),
                                    conv_item_storage.clone(),
                                    conv_id,
                                    original_request.clone(),
                                    response_json.clone(),
                                )
                                .await
                                {
                                    warn!(
                                        "Failed to persist conversation items (stream + MCP): {}",
                                        err
                                    );
                                }
                            }
                        }
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                    return;
                }

                // Execute tools
                let pending_calls = handler.take_pending_calls();

                // Check iteration limit
                state.iteration += 1;
                state.total_calls += pending_calls.len();

                let effective_limit = match max_tool_calls {
                    Some(user_max) => user_max.min(loop_config.max_iterations),
                    None => loop_config.max_iterations,
                };

                if state.total_calls > effective_limit {
                    warn!(
                        "Reached tool call limit during streaming: {}",
                        effective_limit
                    );
                    let error_event = "event: error\ndata: {\"error\": {\"message\": \"Exceeded max_tool_calls limit\"}}\n\n".to_string();
                    let _ = tx.send(Ok(Bytes::from(error_event)));
                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                    return;
                }

                // Execute all pending tool calls
                if !Self::execute_streaming_tool_calls(
                    pending_calls,
                    &active_mcp_clone,
                    &tx,
                    &mut state,
                    server_label,
                    &mut sequence_number,
                )
                .await
                {
                    // Client disconnected during tool execution
                    return;
                }

                // Build resume payload
                match Self::build_resume_payload(
                    &base_payload,
                    &state.conversation_history,
                    &state.original_input,
                    &tools_json,
                    true, // is_streaming = true
                ) {
                    Ok(resume_payload) => {
                        current_payload = resume_payload;
                        // Mark that we're no longer on the first iteration
                        is_first_iteration = false;
                        // Continue loop to make next streaming request
                    }
                    Err(e) => {
                        let error_event = format!("event: error\ndata: {{\"error\": {{\"message\": \"Failed to build resume payload: {}\"}}}}\n\n", e);
                        let _ = tx.send(Ok(Bytes::from(error_event)));
                        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                        return;
                    }
                }
            }
        });

        let body_stream = UnboundedReceiverStream::new(rx);
        let mut response = Response::new(Body::from_stream(body_stream));
        *response.status_mut() = StatusCode::OK;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        response
    }

    /// Parse an SSE block into event name and data
    ///
    /// Returns borrowed strings when possible to avoid allocations in hot paths.
    /// Only allocates when multiple data lines need to be joined.
    fn parse_sse_block(block: &str) -> (Option<&str>, Cow<'_, str>) {
        let mut event_name: Option<&str> = None;
        let mut data_lines: Vec<&str> = Vec::new();

        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start());
            }
        }

        let data = if data_lines.len() == 1 {
            Cow::Borrowed(data_lines[0])
        } else {
            Cow::Owned(data_lines.join("\n"))
        };

        (event_name, data)
    }

    // Note: transform_streaming_event has been replaced by apply_event_transformations_inplace
    // which is more efficient (parses JSON only once instead of twice)

    /// Send mcp_list_tools events to client at the start of streaming
    /// Returns false if client disconnected
    fn send_mcp_list_tools_events(
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        mcp: &Arc<crate::mcp::McpClientManager>,
        server_label: &str,
        output_index: usize,
        sequence_number: &mut u64,
    ) -> bool {
        let tools_item_full = Self::build_mcp_list_tools_item(mcp, server_label);
        let item_id = tools_item_full
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Create empty tools version for the initial added event
        let mut tools_item_empty = tools_item_full.clone();
        if let Some(obj) = tools_item_empty.as_object_mut() {
            obj.insert("tools".to_string(), json!([]));
        }

        // Event 1: response.output_item.added with empty tools
        let event1_payload = json!({
            "type": event_types::OUTPUT_ITEM_ADDED,
            "sequence_number": *sequence_number,
            "output_index": output_index,
            "item": tools_item_empty
        });
        *sequence_number += 1;
        let event1 = format!(
            "event: {}\ndata: {}\n\n",
            event_types::OUTPUT_ITEM_ADDED,
            event1_payload
        );
        if tx.send(Ok(Bytes::from(event1))).is_err() {
            return false; // Client disconnected
        }

        // Event 2: response.mcp_list_tools.in_progress
        let event2_payload = json!({
            "type": event_types::MCP_LIST_TOOLS_IN_PROGRESS,
            "sequence_number": *sequence_number,
            "output_index": output_index,
            "item_id": item_id
        });
        *sequence_number += 1;
        let event2 = format!(
            "event: {}\ndata: {}\n\n",
            event_types::MCP_LIST_TOOLS_IN_PROGRESS,
            event2_payload
        );
        if tx.send(Ok(Bytes::from(event2))).is_err() {
            return false;
        }

        // Event 3: response.mcp_list_tools.completed
        let event3_payload = json!({
            "type": event_types::MCP_LIST_TOOLS_COMPLETED,
            "sequence_number": *sequence_number,
            "output_index": output_index,
            "item_id": item_id
        });
        *sequence_number += 1;
        let event3 = format!(
            "event: {}\ndata: {}\n\n",
            event_types::MCP_LIST_TOOLS_COMPLETED,
            event3_payload
        );
        if tx.send(Ok(Bytes::from(event3))).is_err() {
            return false;
        }

        // Event 4: response.output_item.done with full tools list
        let event4_payload = json!({
            "type": event_types::OUTPUT_ITEM_DONE,
            "sequence_number": *sequence_number,
            "output_index": output_index,
            "item": tools_item_full
        });
        *sequence_number += 1;
        let event4 = format!(
            "event: {}\ndata: {}\n\n",
            event_types::OUTPUT_ITEM_DONE,
            event4_payload
        );
        tx.send(Ok(Bytes::from(event4))).is_ok()
    }

    /// Send mcp_call completion events after tool execution
    /// Returns false if client disconnected
    fn send_mcp_call_completion_events_with_error(
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        call: &FunctionCallInProgress,
        output: &str,
        server_label: &str,
        success: bool,
        error_msg: Option<&str>,
        sequence_number: &mut u64,
    ) -> bool {
        let effective_output_index = call.effective_output_index();

        // Build mcp_call item (reuse existing function)
        let mcp_call_item = Self::build_mcp_call_item(
            &call.name,
            &call.arguments_buffer,
            output,
            server_label,
            success,
            error_msg,
        );

        // Get the mcp_call item_id
        let item_id = mcp_call_item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Event 1: response.mcp_call.completed
        let completed_payload = json!({
            "type": event_types::MCP_CALL_COMPLETED,
            "sequence_number": *sequence_number,
            "output_index": effective_output_index,
            "item_id": item_id
        });
        *sequence_number += 1;

        let completed_event = format!(
            "event: {}\ndata: {}\n\n",
            event_types::MCP_CALL_COMPLETED,
            completed_payload
        );
        if tx.send(Ok(Bytes::from(completed_event))).is_err() {
            return false;
        }

        // Event 2: response.output_item.done (with completed mcp_call)
        let done_payload = json!({
            "type": event_types::OUTPUT_ITEM_DONE,
            "sequence_number": *sequence_number,
            "output_index": effective_output_index,
            "item": mcp_call_item
        });
        *sequence_number += 1;

        let done_event = format!(
            "event: {}\ndata: {}\n\n",
            event_types::OUTPUT_ITEM_DONE,
            done_payload
        );
        tx.send(Ok(Bytes::from(done_event))).is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn send_final_response_event(
        handler: &StreamingToolHandler,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        sequence_number: &mut u64,
        state: &ToolLoopState,
        active_mcp: Option<&Arc<crate::mcp::McpClientManager>>,
        original_request: &ResponsesRequest,
        previous_response_id: Option<&str>,
        server_label: &str,
    ) -> bool {
        let mut final_response = match handler.snapshot_final_response() {
            Some(resp) => resp,
            None => {
                warn!("Final response snapshot unavailable; skipping synthetic completion event");
                return true;
            }
        };

        if let Some(original_id) = handler.original_response_id() {
            if let Some(obj) = final_response.as_object_mut() {
                obj.insert("id".to_string(), Value::String(original_id.to_string()));
            }
        }

        if let Some(mcp) = active_mcp {
            Self::inject_mcp_metadata_streaming(&mut final_response, state, mcp, server_label);
        }

        Self::mask_tools_as_mcp(&mut final_response, original_request);
        Self::patch_streaming_response_json(
            &mut final_response,
            original_request,
            previous_response_id,
        );

        if let Some(obj) = final_response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));
        }

        let completed_payload = json!({
            "type": event_types::RESPONSE_COMPLETED,
            "sequence_number": *sequence_number,
            "response": final_response
        });
        *sequence_number += 1;

        let completed_event = format!(
            "event: {}\ndata: {}\n\n",
            event_types::RESPONSE_COMPLETED,
            completed_payload
        );
        tx.send(Ok(Bytes::from(completed_event))).is_ok()
    }

    /// Inject MCP metadata into a streaming response
    fn inject_mcp_metadata_streaming(
        response: &mut Value,
        state: &ToolLoopState,
        mcp: &Arc<crate::mcp::McpClientManager>,
        server_label: &str,
    ) {
        if let Some(output_array) = response.get_mut("output").and_then(|v| v.as_array_mut()) {
            output_array.retain(|item| {
                item.get("type").and_then(|t| t.as_str())
                    != Some(event_types::ITEM_TYPE_MCP_LIST_TOOLS)
            });

            let list_tools_item = Self::build_mcp_list_tools_item(mcp, server_label);
            output_array.insert(0, list_tools_item);

            let mcp_call_items =
                Self::build_executed_mcp_call_items(&state.conversation_history, server_label);
            let mut insert_pos = 1;
            for item in mcp_call_items {
                output_array.insert(insert_pos, item);
                insert_pos += 1;
            }
        } else if let Some(obj) = response.as_object_mut() {
            let mut output_items = Vec::new();
            output_items.push(Self::build_mcp_list_tools_item(mcp, server_label));
            output_items.extend(Self::build_executed_mcp_call_items(
                &state.conversation_history,
                server_label,
            ));
            obj.insert("output".to_string(), Value::Array(output_items));
        }
    }

    async fn store_response_internal(
        &self,
        response_json: &Value,
        original_body: &ResponsesRequest,
    ) -> Result<(), String> {
        if !original_body.store {
            return Ok(());
        }

        match Self::store_response_impl(&self.response_storage, response_json, original_body).await
        {
            Ok(response_id) => {
                info!(response_id = %response_id.0, "Stored response locally");
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    async fn store_response_impl(
        response_storage: &SharedResponseStorage,
        response_json: &Value,
        original_body: &ResponsesRequest,
    ) -> Result<ResponseId, String> {
        let input_text = match &original_body.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(_) => "complex input".to_string(),
        };

        let output_text = Self::extract_primary_output_text(response_json).unwrap_or_default();

        let mut stored_response = StoredResponse::new(input_text, output_text, None);

        stored_response.instructions = response_json
            .get("instructions")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| original_body.instructions.clone());

        stored_response.model = response_json
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| original_body.model.clone());

        stored_response.user = response_json
            .get("user")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| original_body.user.clone());

        // Set conversation id from request if provided
        if let Some(conv_id) = original_body.conversation.clone() {
            stored_response.conversation_id = Some(conv_id);
        }

        stored_response.metadata = response_json
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_else(|| original_body.metadata.clone().unwrap_or_default());

        stored_response.previous_response_id = response_json
            .get("previous_response_id")
            .and_then(|v| v.as_str())
            .map(ResponseId::from)
            .or_else(|| {
                original_body
                    .previous_response_id
                    .as_ref()
                    .map(|id| ResponseId::from(id.as_str()))
            });

        if let Some(id_str) = response_json.get("id").and_then(|v| v.as_str()) {
            stored_response.id = ResponseId::from(id_str);
        }

        stored_response.raw_response = response_json.clone();

        response_storage
            .store_response(stored_response)
            .await
            .map_err(|e| format!("Failed to store response: {}", e))
    }

    fn patch_streaming_response_json(
        response_json: &mut Value,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<&str>,
    ) {
        if let Some(obj) = response_json.as_object_mut() {
            if let Some(prev_id) = original_previous_response_id {
                let should_insert = obj
                    .get("previous_response_id")
                    .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                    .unwrap_or(true);
                if should_insert {
                    obj.insert(
                        "previous_response_id".to_string(),
                        Value::String(prev_id.to_string()),
                    );
                }
            }

            if !obj.contains_key("instructions")
                || obj
                    .get("instructions")
                    .map(|v| v.is_null())
                    .unwrap_or(false)
            {
                if let Some(instructions) = &original_body.instructions {
                    obj.insert(
                        "instructions".to_string(),
                        Value::String(instructions.clone()),
                    );
                }
            }

            if !obj.contains_key("metadata")
                || obj.get("metadata").map(|v| v.is_null()).unwrap_or(false)
            {
                if let Some(metadata) = &original_body.metadata {
                    let metadata_map: serde_json::Map<String, Value> = metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    obj.insert("metadata".to_string(), Value::Object(metadata_map));
                }
            }

            obj.insert("store".to_string(), Value::Bool(original_body.store));

            if obj
                .get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.is_empty())
                .unwrap_or(true)
            {
                if let Some(model) = &original_body.model {
                    obj.insert("model".to_string(), Value::String(model.clone()));
                }
            }

            if obj.get("user").map(|v| v.is_null()).unwrap_or(false) {
                if let Some(user) = &original_body.user {
                    obj.insert("user".to_string(), Value::String(user.clone()));
                }
            }

            // Attach conversation id for client response if present (final aggregated JSON)
            if let Some(conv_id) = original_body.conversation.clone() {
                obj.insert("conversation".to_string(), json!({"id": conv_id}));
            }
        }
    }

    fn rewrite_streaming_block(
        block: &str,
        original_body: &ResponsesRequest,
        original_previous_response_id: Option<&str>,
    ) -> Option<String> {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut data_lines: Vec<String> = Vec::new();

        for line in trimmed.lines() {
            if line.starts_with("data:") {
                data_lines.push(line.trim_start_matches("data:").trim_start().to_string());
            }
        }

        if data_lines.is_empty() {
            return None;
        }

        let payload = data_lines.join("\n");
        let mut parsed: Value = match serde_json::from_str(&payload) {
            Ok(value) => value,
            Err(err) => {
                warn!("Failed to parse streaming JSON payload: {}", err);
                return None;
            }
        };

        let event_type = parsed
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let should_patch = matches!(
            event_type,
            event_types::RESPONSE_CREATED
                | event_types::RESPONSE_IN_PROGRESS
                | event_types::RESPONSE_COMPLETED
        );

        if !should_patch {
            return None;
        }

        let mut changed = false;
        if let Some(response_obj) = parsed.get_mut("response").and_then(|v| v.as_object_mut()) {
            let desired_store = Value::Bool(original_body.store);
            if response_obj.get("store") != Some(&desired_store) {
                response_obj.insert("store".to_string(), desired_store);
                changed = true;
            }

            if let Some(prev_id) = original_previous_response_id {
                let needs_previous = response_obj
                    .get("previous_response_id")
                    .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                    .unwrap_or(true);

                if needs_previous {
                    response_obj.insert(
                        "previous_response_id".to_string(),
                        Value::String(prev_id.to_string()),
                    );
                    changed = true;
                }
            }

            // Attach conversation id into streaming event response content with ordering
            if let Some(conv_id) = original_body.conversation.clone() {
                response_obj.insert("conversation".to_string(), json!({"id": conv_id}));
                changed = true;
            }
        }

        if !changed {
            return None;
        }

        let new_payload = match serde_json::to_string(&parsed) {
            Ok(json) => json,
            Err(err) => {
                warn!("Failed to serialize modified streaming payload: {}", err);
                return None;
            }
        };

        let mut rebuilt_lines = Vec::new();
        let mut data_written = false;
        for line in trimmed.lines() {
            if line.starts_with("data:") {
                if !data_written {
                    rebuilt_lines.push(format!("data: {}", new_payload));
                    data_written = true;
                }
            } else {
                rebuilt_lines.push(line.to_string());
            }
        }

        if !data_written {
            rebuilt_lines.push(format!("data: {}", new_payload));
        }

        Some(rebuilt_lines.join("\n"))
    }
    fn extract_primary_output_text(response_json: &Value) -> Option<String> {
        if let Some(items) = response_json.get("output").and_then(|v| v.as_array()) {
            for item in items {
                if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                    for part in content {
                        if part
                            .get("type")
                            .and_then(|v| v.as_str())
                            .map(|t| t == "output_text")
                            .unwrap_or(false)
                        {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                return Some(text.to_string());
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

impl OpenAIRouter {
    fn extract_function_call(resp: &Value) -> Option<(String, String, String)> {
        let output = resp.get("output")?.as_array()?;
        for item in output {
            let obj = item.as_object()?;
            let t = obj.get("type")?.as_str()?;
            if t == event_types::ITEM_TYPE_FUNCTION_TOOL_CALL
                || t == event_types::ITEM_TYPE_FUNCTION_CALL
            {
                let call_id = obj
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        obj.get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })?;
                let name = obj.get("name")?.as_str()?.to_string();
                let arguments = obj.get("arguments")?.as_str()?.to_string();
                return Some((call_id, name, arguments));
            }
        }
        None
    }

    /// Replace returned tools with the original request's MCP tool block (if present) so
    /// external clients see MCP semantics rather than internal function tools.
    /// Build MCP tools array value without cloning entire response object
    fn build_mcp_tools_value(original_body: &ResponsesRequest) -> Option<Value> {
        let mcp_tool = original_body
            .tools
            .iter()
            .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some())?;

        let mut m = serde_json::Map::new();
        m.insert("type".to_string(), Value::String("mcp".to_string()));
        if let Some(label) = &mcp_tool.server_label {
            m.insert("server_label".to_string(), Value::String(label.clone()));
        }
        if let Some(url) = &mcp_tool.server_url {
            m.insert("server_url".to_string(), Value::String(url.clone()));
        }
        if let Some(desc) = &mcp_tool.server_description {
            m.insert(
                "server_description".to_string(),
                Value::String(desc.clone()),
            );
        }
        if let Some(req) = &mcp_tool.require_approval {
            m.insert("require_approval".to_string(), Value::String(req.clone()));
        }
        if let Some(allowed) = &mcp_tool.allowed_tools {
            m.insert(
                "allowed_tools".to_string(),
                Value::Array(allowed.iter().map(|s| Value::String(s.clone())).collect()),
            );
        }

        Some(Value::Array(vec![Value::Object(m)]))
    }

    fn mask_tools_as_mcp(resp: &mut Value, original_body: &ResponsesRequest) {
        let mcp_tool = original_body
            .tools
            .iter()
            .find(|t| matches!(t.r#type, ResponseToolType::Mcp) && t.server_url.is_some());
        let Some(t) = mcp_tool else {
            return;
        };

        let mut m = serde_json::Map::new();
        m.insert("type".to_string(), Value::String("mcp".to_string()));
        if let Some(label) = &t.server_label {
            m.insert("server_label".to_string(), Value::String(label.clone()));
        }
        if let Some(url) = &t.server_url {
            m.insert("server_url".to_string(), Value::String(url.clone()));
        }
        if let Some(desc) = &t.server_description {
            m.insert(
                "server_description".to_string(),
                Value::String(desc.clone()),
            );
        }
        if let Some(req) = &t.require_approval {
            m.insert("require_approval".to_string(), Value::String(req.clone()));
        }
        if let Some(allowed) = &t.allowed_tools {
            m.insert(
                "allowed_tools".to_string(),
                Value::Array(allowed.iter().map(|s| Value::String(s.clone())).collect()),
            );
        }

        if let Some(obj) = resp.as_object_mut() {
            obj.insert("tools".to_string(), Value::Array(vec![Value::Object(m)]));
            obj.entry("tool_choice")
                .or_insert(Value::String("auto".to_string()));
        }
    }

    async fn execute_mcp_call(
        mcp_mgr: &Arc<crate::mcp::McpClientManager>,
        tool_name: &str,
        args_json_str: &str,
    ) -> Result<(String, String), String> {
        let args_value: Value =
            serde_json::from_str(args_json_str).map_err(|e| format!("parse tool args: {}", e))?;
        let args_obj = args_value.as_object().cloned();

        let server_name = mcp_mgr
            .get_tool(tool_name)
            .map(|t| t.server)
            .ok_or_else(|| format!("tool not found: {}", tool_name))?;

        let result = mcp_mgr
            .call_tool(tool_name, args_obj)
            .await
            .map_err(|e| format!("tool call failed: {}", e))?;

        let output_str = serde_json::to_string(&result)
            .map_err(|e| format!("Failed to serialize tool result: {}", e))?;
        Ok((server_name, output_str))
    }

    /// Build a resume payload with conversation history
    fn build_resume_payload(
        base_payload: &Value,
        conversation_history: &[Value],
        original_input: &ResponseInput,
        tools_json: &Value,
        is_streaming: bool,
    ) -> Result<Value, String> {
        // Clone the base payload which already has cleaned fields
        let mut payload = base_payload.clone();

        let obj = payload
            .as_object_mut()
            .ok_or_else(|| "payload not an object".to_string())?;

        // Build input array: start with original user input
        let mut input_array = Vec::new();

        // Add original user message
        // For structured input, serialize the original input items
        match original_input {
            ResponseInput::Text(text) => {
                let user_item = json!({
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": text }]
                });
                input_array.push(user_item);
            }
            ResponseInput::Items(items) => {
                // Items are already structured ResponseInputOutputItem, convert to JSON
                if let Ok(items_value) = to_value(items) {
                    if let Some(items_arr) = items_value.as_array() {
                        input_array.extend_from_slice(items_arr);
                    }
                }
            }
        }

        // Add all conversation history (function calls and outputs)
        input_array.extend_from_slice(conversation_history);

        obj.insert("input".to_string(), Value::Array(input_array));

        // Use the transformed tools (function tools, not MCP tools)
        if let Some(tools_arr) = tools_json.as_array() {
            if !tools_arr.is_empty() {
                obj.insert("tools".to_string(), tools_json.clone());
            }
        }

        // Set streaming mode based on caller's context
        obj.insert("stream".to_string(), Value::Bool(is_streaming));
        obj.insert("store".to_string(), Value::Bool(false));

        // Note: SGLang-specific fields were already removed from base_payload
        // before it was passed to execute_tool_loop (see route_responses lines 1935-1946)

        Ok(payload)
    }

    /// Helper function to build mcp_call items from executed tool calls in conversation history
    fn build_executed_mcp_call_items(
        conversation_history: &[Value],
        server_label: &str,
    ) -> Vec<Value> {
        let mut mcp_call_items = Vec::new();

        for item in conversation_history {
            if item.get("type").and_then(|t| t.as_str())
                == Some(event_types::ITEM_TYPE_FUNCTION_CALL)
            {
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");

                // Find corresponding output
                let output_item = conversation_history.iter().find(|o| {
                    o.get("type").and_then(|t| t.as_str()) == Some("function_call_output")
                        && o.get("call_id").and_then(|c| c.as_str()) == Some(call_id)
                });

                let output_str = output_item
                    .and_then(|o| o.get("output").and_then(|v| v.as_str()))
                    .unwrap_or("{}");

                // Check if output contains error by parsing JSON
                let is_error = serde_json::from_str::<Value>(output_str)
                    .map(|v| v.get("error").is_some())
                    .unwrap_or(false);

                let mcp_call_item = Self::build_mcp_call_item(
                    tool_name,
                    args,
                    output_str,
                    server_label,
                    !is_error,
                    if is_error {
                        Some("Tool execution failed")
                    } else {
                        None
                    },
                );
                mcp_call_items.push(mcp_call_item);
            }
        }

        mcp_call_items
    }

    /// Build an incomplete response when limits are exceeded
    fn build_incomplete_response(
        mut response: Value,
        state: ToolLoopState,
        reason: &str,
        active_mcp: &Arc<crate::mcp::McpClientManager>,
        original_body: &ResponsesRequest,
    ) -> Result<Value, String> {
        let obj = response
            .as_object_mut()
            .ok_or_else(|| "response not an object".to_string())?;

        // Set status to completed (not failed - partial success)
        obj.insert("status".to_string(), Value::String("completed".to_string()));

        // Set incomplete_details
        obj.insert(
            "incomplete_details".to_string(),
            json!({ "reason": reason }),
        );

        // Convert any function_call in output to mcp_call format
        if let Some(output_array) = obj.get_mut("output").and_then(|v| v.as_array_mut()) {
            let server_label = original_body
                .tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.as_deref())
                .unwrap_or("mcp");

            // Find any function_call items and convert them to mcp_call (incomplete)
            let mut mcp_call_items = Vec::new();
            for item in output_array.iter() {
                if item.get("type").and_then(|t| t.as_str())
                    == Some(event_types::ITEM_TYPE_FUNCTION_TOOL_CALL)
                {
                    let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");

                    // Mark as incomplete - not executed
                    let mcp_call_item = Self::build_mcp_call_item(
                        tool_name,
                        args,
                        "", // No output - wasn't executed
                        server_label,
                        false, // Not successful
                        Some("Not executed - response stopped due to limit"),
                    );
                    mcp_call_items.push(mcp_call_item);
                }
            }

            // Add mcp_list_tools and executed mcp_call items at the beginning
            if state.total_calls > 0 || !mcp_call_items.is_empty() {
                let list_tools_item = Self::build_mcp_list_tools_item(active_mcp, server_label);
                output_array.insert(0, list_tools_item);

                // Add mcp_call items for executed calls using helper
                let executed_items =
                    Self::build_executed_mcp_call_items(&state.conversation_history, server_label);

                let mut insert_pos = 1;
                for item in executed_items {
                    output_array.insert(insert_pos, item);
                    insert_pos += 1;
                }

                // Add incomplete mcp_call items
                for item in mcp_call_items {
                    output_array.insert(insert_pos, item);
                    insert_pos += 1;
                }
            }
        }

        // Add warning to metadata
        if let Some(metadata_val) = obj.get_mut("metadata") {
            if let Some(metadata_obj) = metadata_val.as_object_mut() {
                if let Some(mcp_val) = metadata_obj.get_mut("mcp") {
                    if let Some(mcp_obj) = mcp_val.as_object_mut() {
                        mcp_obj.insert(
                            "truncation_warning".to_string(),
                            Value::String(format!(
                                "Loop terminated at {} iterations, {} total calls (reason: {})",
                                state.iteration, state.total_calls, reason
                            )),
                        );
                    }
                }
            }
        }

        Ok(response)
    }

    /// Execute the tool calling loop
    async fn execute_tool_loop(
        &self,
        url: &str,
        headers: Option<&HeaderMap>,
        initial_payload: Value,
        original_body: &ResponsesRequest,
        active_mcp: &Arc<crate::mcp::McpClientManager>,
        config: &McpLoopConfig,
    ) -> Result<Value, String> {
        let mut state = ToolLoopState::new(original_body.input.clone());

        // Get max_tool_calls from request (None means no user-specified limit)
        let max_tool_calls = original_body.max_tool_calls.map(|n| n as usize);

        // Keep initial_payload as base template (already has fields cleaned)
        let base_payload = initial_payload.clone();
        let tools_json = base_payload.get("tools").cloned().unwrap_or(json!([]));
        let mut current_payload = initial_payload;

        info!(
            "Starting tool loop: max_tool_calls={:?}, max_iterations={}",
            max_tool_calls, config.max_iterations
        );

        loop {
            // Make request to upstream
            let request_builder = self.client.post(url).json(&current_payload);
            let request_builder = if let Some(headers) = headers {
                apply_request_headers(headers, request_builder, true)
            } else {
                request_builder
            };

            let response = request_builder
                .send()
                .await
                .map_err(|e| format!("upstream request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(format!("upstream error {}: {}", status, body));
            }

            let mut response_json = response
                .json::<Value>()
                .await
                .map_err(|e| format!("parse response: {}", e))?;

            // Check for function call
            if let Some((call_id, tool_name, args_json_str)) =
                Self::extract_function_call(&response_json)
            {
                state.iteration += 1;
                state.total_calls += 1;

                info!(
                    "Tool loop iteration {}: calling {} (call_id: {})",
                    state.iteration, tool_name, call_id
                );

                // Check combined limit: use minimum of user's max_tool_calls (if set) and safety max_iterations
                let effective_limit = match max_tool_calls {
                    Some(user_max) => user_max.min(config.max_iterations),
                    None => config.max_iterations,
                };

                if state.total_calls > effective_limit {
                    if let Some(user_max) = max_tool_calls {
                        if state.total_calls > user_max {
                            warn!("Reached user-specified max_tool_calls limit: {}", user_max);
                        } else {
                            warn!(
                                "Reached safety max_iterations limit: {}",
                                config.max_iterations
                            );
                        }
                    } else {
                        warn!(
                            "Reached safety max_iterations limit: {}",
                            config.max_iterations
                        );
                    }

                    return Self::build_incomplete_response(
                        response_json,
                        state,
                        "max_tool_calls",
                        active_mcp,
                        original_body,
                    );
                }

                // Execute tool
                let call_result =
                    Self::execute_mcp_call(active_mcp, &tool_name, &args_json_str).await;

                let output_str = match call_result {
                    Ok((_, output)) => output,
                    Err(err) => {
                        warn!("Tool execution failed: {}", err);
                        // Return error as output, let model decide how to proceed
                        json!({ "error": err }).to_string()
                    }
                };

                // Record the call
                state.record_call(call_id, tool_name, args_json_str, output_str);

                // Build resume payload
                current_payload = Self::build_resume_payload(
                    &base_payload,
                    &state.conversation_history,
                    &state.original_input,
                    &tools_json,
                    false, // is_streaming = false (non-streaming tool loop)
                )?;
            } else {
                // No more tool calls, we're done
                info!(
                    "Tool loop completed: {} iterations, {} total calls",
                    state.iteration, state.total_calls
                );

                // Inject MCP output items if we executed any tools
                if state.total_calls > 0 {
                    let server_label = original_body
                        .tools
                        .iter()
                        .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                        .and_then(|t| t.server_label.as_deref())
                        .unwrap_or("mcp");

                    // Build mcp_list_tools item
                    let list_tools_item = Self::build_mcp_list_tools_item(active_mcp, server_label);

                    // Insert at beginning of output array
                    if let Some(output_array) = response_json
                        .get_mut("output")
                        .and_then(|v| v.as_array_mut())
                    {
                        output_array.insert(0, list_tools_item);

                        // Build mcp_call items using helper function
                        let mcp_call_items = Self::build_executed_mcp_call_items(
                            &state.conversation_history,
                            server_label,
                        );

                        // Insert mcp_call items after mcp_list_tools using mutable position
                        let mut insert_pos = 1;
                        for item in mcp_call_items {
                            output_array.insert(insert_pos, item);
                            insert_pos += 1;
                        }
                    }
                }

                return Ok(response_json);
            }
        }
    }

    /// Generate a unique ID for MCP output items (similar to OpenAI format)
    fn generate_mcp_id(prefix: &str) -> String {
        use rand::RngCore;
        let mut rng = rand::rng();
        let mut bytes = [0u8; 30];
        rng.fill_bytes(&mut bytes);
        let hex_string: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        format!("{}_{}", prefix, hex_string)
    }

    /// Build an mcp_list_tools output item
    fn build_mcp_list_tools_item(
        mcp: &Arc<crate::mcp::McpClientManager>,
        server_label: &str,
    ) -> Value {
        let tools = mcp.list_tools();
        let tools_json: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters.clone().unwrap_or_else(|| json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    })),
                    "annotations": {
                        "read_only": false
                    }
                })
            })
            .collect();

        json!({
            "id": Self::generate_mcp_id("mcpl"),
            "type": event_types::ITEM_TYPE_MCP_LIST_TOOLS,
            "server_label": server_label,
            "tools": tools_json
        })
    }

    /// Build an mcp_call output item
    fn build_mcp_call_item(
        tool_name: &str,
        arguments: &str,
        output: &str,
        server_label: &str,
        success: bool,
        error: Option<&str>,
    ) -> Value {
        json!({
            "id": Self::generate_mcp_id("mcp"),
            "type": event_types::ITEM_TYPE_MCP_CALL,
            "status": if success { "completed" } else { "failed" },
            "approval_request_id": Value::Null,
            "arguments": arguments,
            "error": error,
            "name": tool_name,
            "output": output,
            "server_label": server_label
        })
    }
}

#[async_trait]
impl super::super::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Simple upstream probe: GET {base}/v1/models without auth
        let url = format!("{}/v1/models", self.base_url);
        match self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) => {
                let code = resp.status();
                // Treat success and auth-required as healthy (endpoint reachable)
                if code.is_success() || code.as_u16() == 401 || code.as_u16() == 403 {
                    (StatusCode::OK, "OK").into_response()
                } else {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("Upstream status: {}", code),
                    )
                        .into_response()
                }
            }
            Err(e) => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Upstream error: {}", e),
            )
                .into_response(),
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let info = json!({
            "router_type": "openai",
            "workers": 1,
            "base_url": &self.base_url
        });
        (StatusCode::OK, info.to_string()).into_response()
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Proxy to upstream /v1/models; forward Authorization header if provided
        let headers = req.headers();

        let mut upstream = self.client.get(format!("{}/v1/models", self.base_url));

        if let Some(auth) = headers
            .get("authorization")
            .or_else(|| headers.get("Authorization"))
        {
            upstream = upstream.header("Authorization", auth);
        }

        match upstream.send().await {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let content_type = res.headers().get(CONTENT_TYPE).cloned();
                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = status;
                        if let Some(ct) = content_type {
                            response.headers_mut().insert(CONTENT_TYPE, ct);
                        }
                        response
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read upstream response: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Failed to contact upstream: {}", e),
            )
                .into_response(),
        }
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        // Not directly supported without model param; return 501
        (
            StatusCode::NOT_IMPLEMENTED,
            "get_model_info not implemented for OpenAI router",
        )
            .into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Generate endpoint is SGLang-specific, not supported for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not supported for OpenAI backend",
        )
            .into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        if !self.circuit_breaker.can_execute() {
            return (StatusCode::SERVICE_UNAVAILABLE, "Circuit breaker open").into_response();
        }

        // Serialize request body, removing SGLang-only fields
        let mut payload = match to_value(body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to serialize request: {}", e),
                )
                    .into_response();
            }
        };
        if let Some(obj) = payload.as_object_mut() {
            for key in [
                "top_k",
                "min_p",
                "min_tokens",
                "regex",
                "ebnf",
                "stop_token_ids",
                "no_stop_trim",
                "ignore_eos",
                "continue_final_message",
                "skip_special_tokens",
                "lora_path",
                "session_params",
                "separate_reasoning",
                "stream_reasoning",
                "chat_template_kwargs",
                "return_hidden_states",
                "repetition_penalty",
                "sampling_seed",
            ] {
                obj.remove(key);
            }
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut req = self.client.post(&url).json(&payload);

        // Forward Authorization header if provided
        if let Some(h) = headers {
            if let Some(auth) = h.get("authorization").or_else(|| h.get("Authorization")) {
                req = req.header("Authorization", auth);
            }
        }

        // Accept SSE when stream=true
        if body.stream {
            req = req.header("Accept", "text/event-stream");
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                self.circuit_breaker.record_failure();
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Failed to contact upstream: {}", e),
                )
                    .into_response();
            }
        };

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !body.stream {
            // Capture Content-Type before consuming response body
            let content_type = resp.headers().get(CONTENT_TYPE).cloned();
            match resp.bytes().await {
                Ok(body) => {
                    self.circuit_breaker.record_success();
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    if let Some(ct) = content_type {
                        response.headers_mut().insert(CONTENT_TYPE, ct);
                    }
                    response
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read response: {}", e),
                    )
                        .into_response()
                }
            }
        } else {
            // Stream SSE bytes to client
            let stream = resp.bytes_stream();
            let (tx, rx) = mpsc::unbounded_channel();
            tokio::spawn(async move {
                let mut s = stream;
                while let Some(chunk) = s.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
            });
            let mut response = Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
            *response.status_mut() = status;
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            response
        }
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Completion endpoint not implemented for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented for OpenAI backend",
        )
            .into_response()
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let url = format!("{}/v1/responses", self.base_url);

        info!(
            requested_store = body.store,
            is_streaming = body.stream,
            "openai_responses_request"
        );

        // Validate mutually exclusive params: previous_response_id and conversation
        // TODO: this validation logic should move the right place, also we need a proper error message module
        if body.previous_response_id.is_some() && body.conversation.is_some() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
                        "type": "invalid_request_error",
                        "param": Value::Null,
                        "code": "mutually_exclusive_parameters"
                    }
                })),
            )
                .into_response();
        }

        // Clone the body and override model if needed
        let mut request_body = body.clone();
        if let Some(model) = model_id {
            request_body.model = Some(model.to_string());
        }
        // Do not forward conversation field upstream; retain for local persistence only
        request_body.conversation = None;

        // Store the original previous_response_id for the response
        let original_previous_response_id = request_body.previous_response_id.clone();

        // Handle previous_response_id by loading prior context
        let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;
        if let Some(prev_id_str) = request_body.previous_response_id.clone() {
            let prev_id = ResponseId::from(prev_id_str.as_str());
            match self
                .response_storage
                .get_response_chain(&prev_id, None)
                .await
            {
                Ok(chain) => {
                    if !chain.responses.is_empty() {
                        let mut items = Vec::new();
                        for stored in chain.responses.iter() {
                            let trimmed_id = stored.id.0.trim_start_matches("resp_");
                            if !stored.input.is_empty() {
                                items.push(ResponseInputOutputItem::Message {
                                    id: format!("msg_u_{}", trimmed_id),
                                    role: "user".to_string(),
                                    status: Some("completed".to_string()),
                                    content: vec![ResponseContentPart::InputText {
                                        text: stored.input.clone(),
                                    }],
                                });
                            }
                            if !stored.output.is_empty() {
                                items.push(ResponseInputOutputItem::Message {
                                    id: format!("msg_a_{}", trimmed_id),
                                    role: "assistant".to_string(),
                                    status: Some("completed".to_string()),
                                    content: vec![ResponseContentPart::OutputText {
                                        text: stored.output.clone(),
                                        annotations: vec![],
                                        logprobs: None,
                                    }],
                                });
                            }
                        }
                        conversation_items = Some(items);
                    } else {
                        info!(previous_response_id = %prev_id_str, "previous chain empty");
                    }
                }
                Err(err) => {
                    warn!(previous_response_id = %prev_id_str, %err, "failed to fetch previous response chain");
                }
            }
            // Clear previous_response_id from request since we're converting to conversation
            request_body.previous_response_id = None;
        }

        // If conversation is provided, attach its items as input to upstream request
        if let Some(conv_id_str) = body.conversation.clone() {
            let conv_id: ConversationId = conv_id_str.as_str().into();
            let mut items: Vec<ResponseInputOutputItem> = Vec::new();
            // Fetch up to MAX_CONVERSATION_HISTORY_ITEMS items in ascending order
            let params = ConversationItemsListParams {
                limit: Self::MAX_CONVERSATION_HISTORY_ITEMS,
                order: ConversationItemsSortOrder::Asc,
                after: None,
            };
            match self
                .conversation_item_storage
                .list_items(&conv_id, params)
                .await
            {
                Ok(stored_items) => {
                    for it in stored_items {
                        match it.item_type.as_str() {
                            "message" => {
                                // content is expected to be an array of ResponseContentPart
                                let parts: Vec<ResponseContentPart> = match serde_json::from_value(
                                    it.content.clone(),
                                ) {
                                    Ok(parts) => parts,
                                    Err(e) => {
                                        warn!(
                                            item_id = %it.id.0,
                                            error = %e,
                                            "Failed to deserialize conversation item content; skipping message item"
                                        );
                                        continue;
                                    }
                                };
                                let role = it.role.unwrap_or_else(|| "user".to_string());
                                items.push(ResponseInputOutputItem::Message {
                                    id: it.id.0,
                                    role,
                                    content: parts,
                                    status: it.status,
                                });
                            }
                            _ => {
                                // Skip unsupported types for request input (e.g., MCP items)
                            }
                        }
                    }
                }
                Err(err) => {
                    warn!(conversation_id = %conv_id.0, error = %err.to_string(), "Failed to load conversation items for request input");
                }
            }

            // Append the current request input at the end
            match &request_body.input {
                ResponseInput::Text(text) => {
                    items.push(ResponseInputOutputItem::Message {
                        id: format!("msg_u_current_{}", items.len()),
                        role: "user".to_string(),
                        status: Some("completed".to_string()),
                        content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    });
                }
                ResponseInput::Items(existing) => {
                    items.extend(existing.clone());
                }
            }
            request_body.input = ResponseInput::Items(items);
        }

        if let Some(mut items) = conversation_items {
            match &request_body.input {
                ResponseInput::Text(text) => {
                    items.push(ResponseInputOutputItem::Message {
                        id: format!("msg_u_current_{}", items.len()),
                        role: "user".to_string(),
                        status: Some("completed".to_string()),
                        content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    });
                }
                ResponseInput::Items(existing) => {
                    items.extend(existing.clone());
                }
            }
            request_body.input = ResponseInput::Items(items);
        }

        // Always set store=false for OpenAI (we store internally)
        request_body.store = false;

        // Convert to JSON payload and strip SGLang-specific fields before forwarding
        let mut payload = match to_value(&request_body) {
            Ok(value) => value,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to serialize responses request: {}", err),
                )
                    .into_response();
            }
        };
        if let Some(obj) = payload.as_object_mut() {
            for key in [
                "request_id",
                "priority",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "top_k",
                "min_p",
                "repetition_penalty",
                "conversation",
            ] {
                obj.remove(key);
            }
        }

        // Check if streaming is requested
        if body.stream {
            // Handle streaming response
            self.handle_streaming_response(
                url,
                headers,
                payload,
                body,
                original_previous_response_id,
            )
            .await
        } else {
            // Handle non-streaming response
            self.handle_non_streaming_response(
                url,
                headers,
                payload,
                body,
                original_previous_response_id,
            )
            .await
        }
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        params: &ResponsesGetParams,
    ) -> Response {
        let stored_id = ResponseId::from(response_id);
        if let Ok(Some(stored_response)) = self.response_storage.get_response(&stored_id).await {
            let stream_requested = params.stream.unwrap_or(false);
            let raw_value = stored_response.raw_response.clone();

            if !raw_value.is_null() {
                if stream_requested {
                    return (
                        StatusCode::NOT_IMPLEMENTED,
                        "Streaming retrieval not yet implemented",
                    )
                        .into_response();
                }

                return (
                    StatusCode::OK,
                    [("content-type", "application/json")],
                    raw_value.to_string(),
                )
                    .into_response();
            }

            let openai_response = ResponsesResponse {
                id: stored_response.id.0.clone(),
                object: "response".to_string(),
                created_at: stored_response.created_at.timestamp(),
                status: ResponseStatus::Completed,
                error: None,
                incomplete_details: None,
                instructions: stored_response.instructions.clone(),
                max_output_tokens: None,
                model: stored_response
                    .model
                    .unwrap_or_else(|| "gpt-4o".to_string()),
                output: vec![ResponseOutputItem::Message {
                    id: format!("msg_{}", stored_response.id.0),
                    role: "assistant".to_string(),
                    status: "completed".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: stored_response.output,
                        annotations: vec![],
                        logprobs: None,
                    }],
                }],
                parallel_tool_calls: true,
                previous_response_id: stored_response.previous_response_id.map(|id| id.0),
                reasoning: None,
                store: true,
                temperature: Some(1.0),
                text: Some(ResponseTextFormat {
                    format: TextFormatType {
                        format_type: "text".to_string(),
                    },
                }),
                tool_choice: "auto".to_string(),
                tools: vec![],
                top_p: Some(1.0),
                truncation: Some("disabled".to_string()),
                usage: None,
                user: stored_response.user.clone(),
                metadata: stored_response.metadata.clone(),
            };

            if stream_requested {
                return (
                    StatusCode::NOT_IMPLEMENTED,
                    "Streaming retrieval not yet implemented",
                )
                    .into_response();
            }

            return (
                StatusCode::OK,
                [("content-type", "application/json")],
                serde_json::to_string(&openai_response).unwrap_or_else(|e| {
                    format!("{{\"error\": \"Failed to serialize response: {}\"}}", e)
                }),
            )
                .into_response();
        }

        (
            StatusCode::NOT_FOUND,
            format!(
                "Response with id '{}' not found in local storage",
                response_id
            ),
        )
            .into_response()
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        // Forward to OpenAI's cancel endpoint
        let url = format!("{}/v1/responses/{}/cancel", self.base_url, response_id);

        let request_builder = self.client.post(&url);

        // Apply headers with filtering (skip content headers for POST without body)
        let request_builder = if let Some(headers) = headers {
            apply_request_headers(headers, request_builder, true)
        } else {
            request_builder
        };

        match request_builder.send().await {
            Ok(response) => {
                let status = response.status();
                let headers = response.headers().clone();

                match response.text().await {
                    Ok(body_text) => {
                        let mut response = (status, body_text).into_response();
                        *response.headers_mut() = preserve_response_headers(&headers);
                        response
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read response body: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Failed to cancel response on OpenAI: {}", e),
            )
                .into_response(),
        }
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::FORBIDDEN,
            "Embeddings endpoint not supported for OpenAI backend",
        )
            .into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::FORBIDDEN,
            "Rerank endpoint not supported for OpenAI backend",
        )
            .into_response()
    }

    async fn create_conversation(&self, _headers: Option<&HeaderMap>, body: &Value) -> Response {
        // TODO: move this spec validation to the right place
        let metadata = match body.get("metadata") {
            Some(Value::Object(map)) => {
                if map.len() > MAX_METADATA_PROPERTIES {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": format!(
                                    "Invalid 'metadata': too many properties. Max {}, got {}",
                                    MAX_METADATA_PROPERTIES, map.len()
                                ),
                                "type": "invalid_request_error",
                                "param": "metadata",
                                "code": "metadata_max_properties_exceeded"
                            }
                        })),
                    )
                        .into_response();
                }
                Some(map.clone())
            }
            Some(Value::Null) | None => None,
            Some(other) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": format!(
                                "Invalid 'metadata': expected object or null but got {}",
                                other
                            ),
                            "type": "invalid_request_error",
                            "param": "metadata",
                            "code": "metadata_invalid_type"
                        }
                    })),
                )
                    .into_response();
            }
        };

        match self
            .conversation_storage
            .create_conversation(crate::data_connector::NewConversation { metadata })
            .await
        {
            Ok(conversation) => {
                (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
            }
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "internal_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
        }
    }

    async fn get_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
    ) -> Response {
        let id: ConversationId = conversation_id.to_string().into();
        match self.conversation_storage.get_conversation(&id).await {
            Ok(Some(conv)) => (StatusCode::OK, Json(conversation_to_json(&conv))).into_response(),
            Ok(None) => (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "message": format!("Conversation with id '{}' not found.", conversation_id),
                        "type": "invalid_request_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "internal_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
        }
    }

    async fn update_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
        body: &Value,
    ) -> Response {
        let id: ConversationId = conversation_id.to_string().into();
        let existing = match self.conversation_storage.get_conversation(&id).await {
            Ok(Some(c)) => c,
            Ok(None) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({
                        "error": {
                            "message": format!("Conversation with id '{}' not found.", conversation_id),
                            "type": "invalid_request_error",
                            "param": Value::Null,
                            "code": Value::Null
                        }
                    })),
                )
                    .into_response();
            }
            Err(err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {
                            "message": err.to_string(),
                            "type": "internal_error",
                            "param": Value::Null,
                            "code": Value::Null
                        }
                    })),
                )
                    .into_response();
            }
        };

        // Parse metadata patch
        enum Patch {
            NoChange,
            ClearAll,
            Merge(ConversationMetadata),
        }
        let patch = match body.get("metadata") {
            None => Patch::NoChange,
            Some(Value::Null) => Patch::ClearAll,
            Some(Value::Object(map)) => Patch::Merge(map.clone()),
            Some(other) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": format!(
                                "Invalid 'metadata': expected object or null but got {}",
                                other
                            ),
                            "type": "invalid_request_error",
                            "param": "metadata",
                            "code": "metadata_invalid_type"
                        }
                    })),
                )
                    .into_response();
            }
        };

        let merged_metadata = match patch {
            Patch::NoChange => {
                return (StatusCode::OK, Json(conversation_to_json(&existing))).into_response();
            }
            Patch::ClearAll => None,
            Patch::Merge(upd) => {
                let mut merged = existing.metadata.clone().unwrap_or_default();
                let previous = merged.len();
                for (k, v) in upd.into_iter() {
                    if v.is_null() {
                        merged.remove(&k);
                    } else {
                        merged.insert(k, v);
                    }
                }
                let updated = merged.len();
                if updated > MAX_METADATA_PROPERTIES {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": format!(
                                    "Invalid 'metadata': too many properties after update. Max {} ({} -> {}).",
                                    MAX_METADATA_PROPERTIES, previous, updated
                                ),
                                "type": "invalid_request_error",
                                "param": "metadata",
                                "code": "metadata_max_properties_exceeded",
                                "extra": {
                                    "previous_property_count": previous,
                                    "updated_property_count": updated
                                }
                            }
                        })),
                    )
                        .into_response();
                }
                if merged.is_empty() {
                    None
                } else {
                    Some(merged)
                }
            }
        };

        match self
            .conversation_storage
            .update_conversation(&id, merged_metadata)
            .await
        {
            Ok(Some(conv)) => (StatusCode::OK, Json(conversation_to_json(&conv))).into_response(),
            Ok(None) => (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "message": format!("Conversation with id '{}' not found.", conversation_id),
                        "type": "invalid_request_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "internal_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
        }
    }

    async fn delete_conversation(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
    ) -> Response {
        let id: ConversationId = conversation_id.to_string().into();
        match self.conversation_storage.delete_conversation(&id).await {
            Ok(true) => (
                StatusCode::OK,
                Json(json!({
                    "id": conversation_id,
                    "object": "conversation.deleted",
                    "deleted": true
                })),
            )
                .into_response(),
            Ok(false) => (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "message": format!("Conversation with id '{}' not found.", conversation_id),
                        "type": "invalid_request_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "internal_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
        }
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }

    async fn list_conversation_items(
        &self,
        _headers: Option<&HeaderMap>,
        conversation_id: &str,
        limit: Option<usize>,
        order: Option<String>,
        after: Option<String>,
    ) -> Response {
        let id: ConversationId = conversation_id.into();
        match self.conversation_storage.get_conversation(&id).await {
            Ok(Some(_)) => {}
            Ok(None) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({
                        "error": {
                            "message": format!("Conversation with id '{}' not found.", conversation_id),
                            "type": "invalid_request_error",
                            "param": Value::Null,
                            "code": Value::Null
                        }
                    })),
                )
                    .into_response();
            }
            Err(err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {
                            "message": err.to_string(),
                            "type": "internal_error",
                            "param": Value::Null,
                            "code": Value::Null
                        }
                    })),
                )
                    .into_response();
            }
        }

        let lim = limit.unwrap_or(20).clamp(1, 100);
        let sort = match order.as_deref() {
            Some("asc") => ConversationItemsSortOrder::Asc,
            _ => ConversationItemsSortOrder::Desc,
        };
        let params = ConversationItemsListParams {
            limit: lim + 1,
            order: sort,
            after,
        };

        match self.conversation_item_storage.list_items(&id, params).await {
            Ok(mut items) => {
                let has_more = items.len() > lim;
                if has_more {
                    items.truncate(lim);
                }
                let data: Vec<Value> = items
                    .into_iter()
                    .map(|it| {
                        json!({
                            "id": it.id.0,
                            "type": it.item_type,
                            "status": it.status.unwrap_or_else(|| "completed".to_string()),
                            "content": it.content,
                            "role": it.role,
                        })
                    })
                    .collect();
                let first_id = data
                    .first()
                    .and_then(|v| v.get("id"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let last_id = data
                    .last()
                    .and_then(|v| v.get("id"))
                    .cloned()
                    .unwrap_or(Value::Null);
                (
                    StatusCode::OK,
                    Json(json!({
                        "object": "list",
                        "data": data,
                        "first_id": first_id,
                        "last_id": last_id,
                        "has_more": has_more
                    })),
                )
                    .into_response()
            }
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "internal_error",
                        "param": Value::Null,
                        "code": Value::Null
                    }
                })),
            )
                .into_response(),
        }
    }
}
// Maximum number of properties allowed in conversation metadata (align with server)
const MAX_METADATA_PROPERTIES: usize = 16;

fn conversation_to_json(conversation: &Conversation) -> Value {
    json!({
        "id": conversation.id.0,
        "object": "conversation",
        "created_at": conversation.created_at.timestamp(),
        "metadata": to_value(&conversation.metadata).unwrap_or(Value::Null),
    })
}

async fn persist_items_with_storages(
    conv_storage: SharedConversationStorage,
    item_storage: SharedConversationItemStorage,
    conversation_id: String,
    request: ResponsesRequest,
    response: Value,
) -> Result<(), String> {
    let conv_id: ConversationId = conversation_id.as_str().into();
    match conv_storage.get_conversation(&conv_id).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            warn!(conversation_id = %conv_id.0, "Conversation not found; skipping item persistence");
            return Ok(());
        }
        Err(err) => return Err(err.to_string()),
    }

    // Extract response_id once for attaching to both input and output items
    let response_id_opt = response
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Helper to ensure status defaults to completed
    async fn create_and_link_item(
        item_storage: &SharedConversationItemStorage,
        conv_id: &ConversationId,
        mut new_item: DCNewConversationItem,
    ) -> Result<(), String> {
        if new_item.status.is_none() {
            new_item.status = Some("completed".to_string());
        }
        let created = item_storage
            .create_item(new_item)
            .await
            .map_err(|e| e.to_string())?;
        item_storage
            .link_item(conv_id, &created.id, chrono::Utc::now())
            .await
            .map_err(|e| e.to_string())?;
        tracing::info!(conversation_id = %conv_id.0, item_id = %created.id.0, item_type = %created.item_type, "Persisted conversation item and link");
        Ok(())
    }

    match request.input.clone() {
        ResponseInput::Text(text) => {
            let new_item = DCNewConversationItem {
                id: None, // generate new message id for input
                response_id: response_id_opt.clone(),
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([{ "type": "input_text", "text": text }]),
                status: Some("completed".to_string()),
            };
            create_and_link_item(&item_storage, &conv_id, new_item).await?;
        }
        ResponseInput::Items(items) => {
            for input_item in items {
                match input_item {
                    ResponseInputOutputItem::Message {
                        role,
                        content,
                        status,
                        ..
                    } => {
                        let content_v =
                            serde_json::to_value(&content).map_err(|e| e.to_string())?;
                        let new_item = DCNewConversationItem {
                            id: None, // generate new id for input items
                            response_id: response_id_opt.clone(),
                            item_type: "message".to_string(),
                            role: Some(role),
                            content: content_v,
                            status,
                        };
                        create_and_link_item(&item_storage, &conv_id, new_item).await?;
                    }
                    ResponseInputOutputItem::Reasoning {
                        summary,
                        content,
                        status,
                        ..
                    } => {
                        let new_item = DCNewConversationItem {
                            id: None, // generate new id for input items
                            response_id: response_id_opt.clone(),
                            item_type: "reasoning".to_string(),
                            role: None,
                            content: json!({ "summary": summary, "content": content }),
                            status,
                        };
                        create_and_link_item(&item_storage, &conv_id, new_item).await?;
                    }
                    ResponseInputOutputItem::FunctionToolCall {
                        name,
                        arguments,
                        output,
                        status,
                        ..
                    } => {
                        let new_item = DCNewConversationItem {
                            id: None, // generate new id for input items
                            response_id: response_id_opt.clone(),
                            item_type: "function_tool_call".to_string(),
                            role: None,
                            content: json!({ "name": name, "arguments": arguments, "output": output }),
                            status,
                        };
                        create_and_link_item(&item_storage, &conv_id, new_item).await?;
                    }
                }
            }
        }
    }

    if let Some(output_array) = response.get("output").and_then(|v| v.as_array()) {
        for item in output_array {
            let item_type = match item.get("type").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => continue,
            };

            match item_type {
                "message" => {
                    let id_in = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| crate::data_connector::ConversationItemId(s.to_string()));
                    let role = item
                        .get("role")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let content_v = item
                        .get("content")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new()));
                    let status = item
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let new_item = DCNewConversationItem {
                        id: id_in,
                        response_id: response_id_opt.clone(),
                        item_type: "message".to_string(),
                        role,
                        content: content_v,
                        status,
                    };
                    create_and_link_item(&item_storage, &conv_id, new_item).await?;
                }
                "reasoning" => {
                    let id_in = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let summary_v = item
                        .get("summary")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new()));
                    let content_v = item
                        .get("content")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new()));
                    let status = item
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let new_item = DCNewConversationItem {
                        id: id_in.map(crate::data_connector::ConversationItemId),
                        response_id: response_id_opt.clone(),
                        item_type: "reasoning".to_string(),
                        role: None,
                        content: json!({ "summary": summary_v, "content": content_v }),
                        status,
                    };
                    create_and_link_item(&item_storage, &conv_id, new_item).await?;
                }
                "function_tool_call" => {
                    let id_in = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                    let output_str = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    let status = item
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let new_item = DCNewConversationItem {
                        id: id_in.map(crate::data_connector::ConversationItemId),
                        response_id: response_id_opt.clone(),
                        item_type: "function_tool_call".to_string(),
                        role: None,
                        content: json!({
                            "name": name,
                            "arguments": arguments,
                            "output": output_str
                        }),
                        status,
                    };
                    create_and_link_item(&item_storage, &conv_id, new_item).await?;
                }
                "mcp_call" => {
                    let id_in = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
                    let output_str = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    let status = item
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let content_v = json!({
                        "server_label": item.get("server_label").cloned().unwrap_or(Value::Null),
                        "name": name,
                        "arguments": arguments,
                        "output": output_str,
                        "error": item.get("error").cloned().unwrap_or(Value::Null),
                        "approval_request_id": item.get("approval_request_id").cloned().unwrap_or(Value::Null)
                    });
                    let new_item = DCNewConversationItem {
                        id: id_in.map(crate::data_connector::ConversationItemId),
                        response_id: response_id_opt.clone(),
                        item_type: "mcp_call".to_string(),
                        role: None,
                        content: content_v,
                        status,
                    };
                    create_and_link_item(&item_storage, &conv_id, new_item).await?;
                }
                "mcp_list_tools" => {
                    let id_in = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let content_v = json!({
                        "server_label": item.get("server_label").cloned().unwrap_or(Value::Null),
                        "tools": item.get("tools").cloned().unwrap_or_else(|| Value::Array(Vec::new()))
                    });
                    let new_item = DCNewConversationItem {
                        id: id_in.map(crate::data_connector::ConversationItemId),
                        response_id: response_id_opt.clone(),
                        item_type: "mcp_list_tools".to_string(),
                        role: None,
                        content: content_v,
                        status: Some("completed".to_string()),
                    };
                    create_and_link_item(&item_storage, &conv_id, new_item).await?;
                }
                _ => {}
            }
        }
    }

    Ok(())
}
