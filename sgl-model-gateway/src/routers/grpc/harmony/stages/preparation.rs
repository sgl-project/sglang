//! Harmony Preparation Stage: Harmony encoding for chat and generate requests

use async_trait::async_trait;
use axum::response::Response;
use serde_json::json;
use tracing::error;

use super::super::HarmonyBuilder;
use crate::{
    protocols::{
        chat::ChatCompletionRequest,
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
    tool_parser::constraints,
};

/// Harmony Preparation stage: Encode requests using Harmony protocol
///
/// Replaces the regular PreparationStage for Harmony models.
/// Converts chat/generate requests to Harmony-encoded token_ids and extraction_text.
pub struct HarmonyPreparationStage {
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
        // Validate - reject logprobs
        if request.logprobs {
            error!(
                function = "prepare_chat",
                "logprobs requested but not supported for Harmony models"
            );
            return Err(error::bad_request(
                "harmony_logprobs_not_supported",
                "logprobs are not supported for Harmony models".to_string(),
            ));
        }

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 2: Build tool constraints
        let tool_constraints = if let Some(tools) = body_ref.tools.as_ref() {
            constraints::build_tool_call_constraint(
                tools,
                &body_ref.tool_choice,
                request.parallel_tool_calls.unwrap_or(true),
                &ctx.components.tool_parser_factory,
                None, // configured_parser not available in context yet
                &request.model,
            )
            .map_err(|e| {
                error!(function = "prepare_chat", error = %e, "Invalid tool configuration");
                error::bad_request("invalid_tool_configuration", format!("Invalid tool configuration: {}", e))
            })?
        } else {
            None
        };

        // Step 3: Build via Harmony
        let build_output = self.builder.build_from_chat(&body_ref).map_err(|e| {
            error!(
                function = "prepare_chat",
                error = %e,
                "Harmony build failed for chat request"
            );
            error::bad_request(
                "harmony_build_failed",
                format!("Harmony build failed: {}", e),
            )
        })?;

        // Step 4: Store results
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages: None,
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
            constraints::build_tool_call_constraint(
                &function_tools,
                &request.tool_choice,
                request.parallel_tool_calls.unwrap_or(true),
                &ctx.components.tool_parser_factory,
                None, // configured_parser not available in context yet
                &request.model,
            )
            .map_err(|e| {
                error!(function = "prepare_responses", error = %e, "Invalid tool configuration");
                error::bad_request("invalid_tool_configuration", format!("Invalid tool configuration: {}", e))
            })?
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
}

/// Build Harmony structural tag for structured output (JSON schema constraint)
///
/// Creates a structural tag that applies JSON schema constraint to the final channel,
/// supporting both reasoning-enabled and reasoning-disabled modes:
/// - With reasoning: triggers on `<|start|>assistant<|channel|>final` (waits for analysis to complete)
/// - Without reasoning: triggers on `<|channel|>final` (goes directly to final channel)
///
/// This is used for the Responses API text.format field (json_object or json_schema).
pub fn build_text_format_structural_tag(schema: &serde_json::Value) -> Result<String, String> {
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
