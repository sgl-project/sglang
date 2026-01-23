//! Harmony Preparation Stage: Harmony encoding for chat and generate requests

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use axum::response::Response;
use serde_json::json;
use tracing::error;

use super::super::HarmonyBuilder;
use crate::{
    multimodal::{
        types::{ChatContentPart, ImageDetail, Modality, TrackedMedia},
        AsyncMultiModalTracker, TrackerConfig,
    },
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        common::{Tool, ToolChoice, ToolChoiceValue},
        responses::ResponsesRequest,
    },
    routers::{
        error,
        grpc::{
            common::{responses::utils::extract_tools_from_response_tools, stages::PipelineStage},
            context::{PreparationOutput, RequestContext, RequestType},
            utils,
        },
    },
};

/// Harmony Preparation stage: Encode requests using Harmony protocol
///
/// Replaces the regular PreparationStage for Harmony models.
/// Converts chat/generate requests to Harmony-encoded token_ids and extraction_text.
pub(crate) struct HarmonyPreparationStage {
    builder: HarmonyBuilder,
}

impl HarmonyPreparationStage {
    /// Create a new Harmony preparation stage
    pub fn new() -> Self {
        Self {
            builder: HarmonyBuilder::new(),
        }
    }
}

impl Default for HarmonyPreparationStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for HarmonyPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Clone Arc before match to avoid borrow checker issues
        // Arc clone is cheap (8 bytes) - avoids full request clone (15KB-200KB)
        let is_chat = matches!(&ctx.input.request_type, RequestType::Chat(_));
        let is_responses = matches!(&ctx.input.request_type, RequestType::Responses(_));

        if is_chat {
            let request_arc = ctx.chat_request_arc();
            self.prepare_chat(ctx, &request_arc).await?;
        } else if is_responses {
            let request_arc = ctx.responses_request_arc();
            self.prepare_responses(ctx, &request_arc).await?;
        } else {
            error!(
                function = "HarmonyPreparationStage::execute",
                "Unsupported request type for Harmony pipeline"
            );
            return Err(error::bad_request(
                "harmony_request_type_invalid",
                "Only Chat and Responses requests supported in Harmony pipeline".to_string(),
            ));
        }

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyPreparation"
    }
}

impl HarmonyPreparationStage {
    /// Prepare a chat completion request using Harmony encoding
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<Option<Response>, Response> {
        // Step 0: Resolve tokenizer and validate
        let tokenizer = utils::resolve_tokenizer(ctx, "HarmonyPreparationStage::prepare_chat")
            .map_err(|e| *e)?;

        if request.logprobs {
            return Err(error::bad_request(
                "harmony_logprobs_not_supported",
                "logprobs are not supported for Harmony models".to_string(),
            ));
        }

        // Step 1: Handle Multimodal Tracking for zero-copy extraction
        let mut multimodal_data = None;
        if request.is_multimodal() {
            let mut tracker = AsyncMultiModalTracker::new(
                ctx.components.media_connector.clone(),
                TrackerConfig::default(),
            );

            for message in &request.messages {
                match message {
                    ChatMessage::User { content, .. }
                    | ChatMessage::System { content, .. }
                    | ChatMessage::Tool { content, .. }
                    | ChatMessage::Developer { content, .. } => {
                        if let MessageContent::Parts(parts) = content {
                            for part in parts {
                                let chat_part = match part {
                                    crate::protocols::common::ContentPart::Text { text } => {
                                        ChatContentPart::Text { text: text.clone() }
                                    }
                                    crate::protocols::common::ContentPart::ImageUrl {
                                        image_url,
                                    } => ChatContentPart::ImageUrl {
                                        url: image_url.url.clone(),
                                        detail: image_url.detail.as_ref().map(|d| {
                                            match d.as_str() {
                                                "low" => ImageDetail::Low,
                                                "high" => ImageDetail::High,
                                                _ => ImageDetail::Auto,
                                            }
                                        }),
                                        uuid: None,
                                    },
                                    crate::protocols::common::ContentPart::VideoUrl { .. } => {
                                        return Err(error::bad_request(
                                            "unsupported_media_type",
                                            "Video inputs are not yet supported in gRPC Harmony chat",
                                        ));
                                    }
                                };
                                tracker.push_part(chat_part).map_err(|e| {
                                    error!(error = %e, "Failed to push multimodal part to tracker (Harmony)");
                                    error::bad_request("multimodal_tracking_failed", e.to_string())
                                })?;
                            }
                        }
                    }
                    ChatMessage::Assistant { content, .. } => {
                        if let Some(MessageContent::Parts(parts)) = content {
                            for part in parts {
                                if let crate::protocols::common::ContentPart::ImageUrl {
                                    image_url,
                                } = part
                                {
                                    tracker
                                        .push_part(ChatContentPart::ImageUrl {
                                            url: image_url.url.clone(),
                                            detail: image_url.detail.as_ref().map(|d| {
                                                match d.as_str() {
                                                    "low" => ImageDetail::Low,
                                                    "high" => ImageDetail::High,
                                                    _ => ImageDetail::Auto,
                                                }
                                            }),
                                            uuid: None,
                                        })
                                        .map_err(|e| {
                                            error::bad_request(
                                                "multimodal_tracking_failed",
                                                e.to_string(),
                                            )
                                        })?;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            let output = tracker.finalize().await.map_err(|e| {
                error!("Multimodal tracking failed in Harmony: {}", e);
                error::internal_error("multimodal_tracking_failed", e.to_string())
            })?;

            if let Some(media_vec) = output.data.get(&Modality::Image) {
                let frames = media_vec
                    .iter()
                    .filter_map(|m| {
                        if let TrackedMedia::Image(frame) = m {
                            Some(frame.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                if !frames.is_empty() {
                    let mut map = HashMap::new();
                    map.insert(Modality::Image, frames);
                    multimodal_data = Some(map);
                }
            }
        }

        // Step 2: Filter tools and process messages for multimodal payload
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Zero-Copy: populate ProcessedMessages to carry multimodal_inputs to gRPC builder
        let processed_messages = match utils::process_chat_messages(
            &body_ref,
            &*tokenizer,
            multimodal_data,
        ) {
            Ok(msgs) => Some(msgs),
            Err(e) => {
                error!(function = "HarmonyPreparationStage::prepare_chat", error = %e, "Failed to process chat messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Step 3: Build tool constraints and Harmony encoding
        let tool_constraints = if let Some(tools) = body_ref.tools.as_ref() {
            Self::generate_tool_call_constraint(tools, &body_ref.tool_choice).map_err(|e| *e)?
        } else {
            None
        };

        let build_output = self.builder.build_from_chat(&body_ref).map_err(|e| {
            error!(function = "prepare_chat", error = %e, "Harmony build failed");
            error::bad_request(
                "harmony_build_failed",
                format!("Harmony build failed: {}", e),
            )
        })?;

        // Step 4: Store results
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages,
            tool_constraints,
            filtered_request: if matches!(body_ref, std::borrow::Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
        });

        Ok(None)
    }

    /// Prepare a responses API request using Harmony encoding
    ///
    /// For responses API, we build from conversation history using the same Harmony
    /// encoding that the builder provides. This handles the MCP loop integration.
    pub async fn prepare_responses(
        &self,
        ctx: &mut RequestContext,
        request: &ResponsesRequest,
    ) -> Result<Option<Response>, Response> {
        // Step 1: Extract function and MCP tools with schemas from ResponseTools
        let mut function_tools = extract_tools_from_response_tools(request.tools.as_deref(), true);

        // Step 2: Filter tools based on tool_choice (AllowedTools or Function)
        // Note: Tool existence is already validated in ResponsesRequest::validate()
        if let Some(filtered) =
            utils::filter_tools_by_tool_choice(&function_tools, &request.tool_choice)
        {
            function_tools = filtered;
        }

        // Step 3: Generate Harmony structural tags
        let tool_constraint = if !function_tools.is_empty() {
            Self::generate_tool_call_constraint(&function_tools, &request.tool_choice)
                .map_err(|e| *e)?
        } else {
            None
        };

        let text_constraint = if let Some(text_config) = &request.text {
            Self::generate_text_format_constraint(text_config).map_err(|e| *e)?
        } else {
            None
        };

        if tool_constraint.is_some() && text_constraint.is_some() {
            error!(
                function = "prepare_responses",
                "Conflicting constraints: both tool_choice and text format specified"
            );
            return Err(error::bad_request(
                "conflicting_constraints",
                "Cannot use both tool_choice (required/function) and text format (json_object/json_schema) simultaneously".to_string(),
            ));
        }

        let constraint = tool_constraint.or(text_constraint);

        // Step 3: Build via Harmony from responses API request
        let build_output = self.builder.build_from_responses(request).map_err(|e| {
            error!(
                function = "prepare_responses",
                error = %e,
                "Harmony build failed for responses request"
            );
            error::bad_request(
                "harmony_build_failed",
                format!("Harmony build failed: {}", e),
            )
        })?;

        // Step 4: Store results with constraint
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages: None,
            tool_constraints: constraint,
            filtered_request: None,
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
        });

        Ok(None)
    }

    /// Generate Harmony structural tag for structured output (text field)
    ///
    /// Converts text.format to structural tag that constrains the final channel.
    /// Returns None if text.format is not specified or is "text".
    fn generate_text_format_constraint(
        text_config: &crate::protocols::responses::TextConfig,
    ) -> Result<Option<(String, String)>, Box<Response>> {
        use crate::protocols::responses::TextFormat;

        let Some(format) = &text_config.format else {
            return Ok(None);
        };

        match format {
            TextFormat::Text => Ok(None),
            TextFormat::JsonObject => {
                let tag = build_text_format_structural_tag(&serde_json::json!({"type": "object"}))
                    .map_err(|e| {
                        error!(
                            function = "generate_text_format_constraint",
                            error = %e,
                            "Failed to build text format structural tag for JsonObject"
                        );
                        Box::new(error::internal_error("build_text_format_tag_failed", e))
                    })?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            TextFormat::JsonSchema { schema, .. } => {
                let tag = build_text_format_structural_tag(schema).map_err(|e| {
                    error!(
                        function = "generate_text_format_constraint",
                        error = %e,
                        "Failed to build text format structural tag for JsonSchema"
                    );
                    Box::new(error::internal_error("build_text_format_tag_failed", e))
                })?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
        }
    }

    /// Generate Harmony structural tag for tool constraints
    ///
    /// Uses structural tags with `triggered_tags` format to force Harmony format output.
    /// This ensures the model outputs in Harmony format (with channels) even when constrained.
    fn generate_tool_call_constraint(
        tools: &[Tool],
        tool_choice: &Option<ToolChoice>,
    ) -> Result<Option<(String, String)>, Box<Response>> {
        let Some(choice) = tool_choice.as_ref() else {
            return Ok(None);
        };

        match choice {
            ToolChoice::Function { function, .. } => {
                let tag = Self::build_tool_call_structural_tag(tools, Some(&function.name))?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            ToolChoice::Value(ToolChoiceValue::Required) => {
                let tag = Self::build_tool_call_structural_tag(tools, None)?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            ToolChoice::AllowedTools { mode, .. } => {
                if mode == "required" {
                    let tag = Self::build_tool_call_structural_tag(tools, None)?;
                    Ok(Some(("structural_tag".to_string(), tag)))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Build Harmony structural tag for tool calling constraints
    ///
    /// Supports both reasoning-enabled and reasoning-disabled modes:
    /// - With reasoning: triggers on `<|start|>assistant<|channel|>commentary` (waits for analysis)
    /// - Without reasoning: triggers on `<|channel|>commentary` (goes directly to commentary)
    fn build_tool_call_structural_tag(
        tools: &[Tool],
        specific_function: Option<&str>,
    ) -> Result<String, Box<Response>> {
        let mut tags = Vec::new();

        // Filter tools if specific function requested
        let tools_to_use: Vec<&Tool> = if let Some(func_name) = specific_function {
            tools
                .iter()
                .filter(|t| t.function.name == func_name)
                .collect()
        } else {
            tools.iter().collect()
        };

        // Validate specific function exists
        match specific_function {
            Some(tool_name) if tools_to_use.is_empty() => {
                error!(
                    function = "generate_tool_call_constraint",
                    tool_name = %tool_name,
                    "Specified tool not found in tools list"
                );
                return Err(Box::new(error::bad_request(
                    "tool_not_found",
                    format!("Tool '{}' not found in tools list", tool_name),
                )));
            }
            _ => {}
        }

        // Build tags for each tool - need two patterns per tool for reasoning on/off
        for tool in tools_to_use {
            let tool_name = &tool.function.name;
            let params_schema = &tool.function.parameters;

            // Pattern 1: For reasoning-enabled mode (with analysis channel before commentary)
            tags.push(json!({
                "begin": format!("<|start|>assistant<|channel|>commentary to=functions.{}<|constrain|>json<|message|>", tool_name),
                "content": {
                    "type": "json_schema",
                    "json_schema": params_schema
                },
                "end": "" // `end` is empty because <|call|> comes naturally from Harmony stop tokens
            }));

            // Pattern 2: For reasoning-disabled mode (goes directly to commentary channel)
            tags.push(json!({
                "begin": format!("<|channel|>commentary to=functions.{}<|constrain|>json<|message|>", tool_name),
                "content": {
                    "type": "json_schema",
                    "json_schema": params_schema
                },
                "end": ""
            }));
        }

        let stop_after_first = specific_function.is_some();

        let structural_tag = json!({
            "format": {
                "type": "triggered_tags",
                "triggers": ["<|start|>assistant<|channel|>commentary", "<|channel|>commentary"],
                "tags": tags,
                "at_least_one": true,
                "stop_after_first": stop_after_first
            }
        });

        serde_json::to_string(&structural_tag).map_err(|e| {
            error!(
                function = "generate_tool_call_constraint",
                error = %e,
                "Failed to serialize structural tag"
            );
            Box::new(error::internal_error(
                "serialize_structural_tag_failed",
                format!("Failed to serialize structural tag: {}", e),
            ))
        })
    }
}

/// Build Harmony structural tag for structured output (JSON schema constraint)
///
/// Creates a structural tag that applies JSON schema constraint to the final channel,
/// supporting both reasoning-enabled and reasoning-disabled modes:
/// - With reasoning: triggers on `<|start|>assistant<|channel|>final` (waits for analysis to complete)
/// - Without reasoning: triggers on `<|channel|>final` (goes directly to final channel)
///
/// This is used for the Responses API text.format field (json_object or json_schema).
pub(crate) fn build_text_format_structural_tag(
    schema: &serde_json::Value,
) -> Result<String, String> {
    let structural_tag = json!({
        "format": {
            "type": "triggered_tags",
            "triggers": ["<|start|>assistant<|channel|>final", "<|channel|>final"],
            "tags": [
                {
                    // Pattern 1: For reasoning-enabled mode (with analysis channel before final)
                    "begin": "<|start|>assistant<|channel|>final<|constrain|>json<|message|>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema
                    },
                    "end": ""
                },
                {
                    // Pattern 2: For reasoning-disabled mode (goes directly to final channel)
                    "begin": "<|channel|>final<|constrain|>json<|message|>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema
                    },
                    "end": ""
                }
            ],
            "at_least_one": true,
            "stop_after_first": true
        }
    });

    serde_json::to_string(&structural_tag).map_err(|e| {
        format!(
            "Failed to serialize structural tag for structured output: {}",
            e
        )
    })
}
