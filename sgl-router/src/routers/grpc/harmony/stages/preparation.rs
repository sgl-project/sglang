//! Harmony Preparation Stage: Harmony encoding for chat and generate requests

use async_trait::async_trait;
use axum::response::Response;
use serde_json::json;

use super::super::HarmonyBuilder;
use crate::{
    protocols::{
        chat::ChatCompletionRequest,
        common::{Tool, ToolChoice, ToolChoiceValue},
        responses::ResponsesRequest,
    },
    routers::grpc::{
        context::{PreparationOutput, RequestContext, RequestType},
        stages::PipelineStage,
        utils,
    },
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
            return Err(utils::bad_request_error(
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
            return Err(utils::bad_request_error(
                "logprobs are not supported for Harmony models".to_string(),
            ));
        }

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_tools_for_request(request);

        // Step 2: Build tool constraints
        let tool_constraints = if let Some(tools) = body_ref.tools.as_ref() {
            Self::generate_harmony_structural_tag(tools, &body_ref.tool_choice).map_err(|e| *e)?
        } else {
            None
        };

        // Step 3: Build via Harmony
        let build_output = self
            .builder
            .build_from_chat(&body_ref)
            .map_err(|e| utils::bad_request_error(format!("Harmony build failed: {}", e)))?;

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
        // Build via Harmony from responses API request
        let build_output = self
            .builder
            .build_from_responses(request)
            .map_err(|e| utils::bad_request_error(format!("Harmony build failed: {}", e)))?;

        // Store results in preparation output
        ctx.state.preparation = Some(PreparationOutput {
            original_text: None,
            token_ids: build_output.input_ids,
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            harmony_mode: true,
            selection_text: Some(build_output.selection_text),
            harmony_messages: Some(build_output.harmony_messages),
            harmony_stop_ids: Some(build_output.stop_token_ids),
        });

        Ok(None)
    }

    /// Generate Harmony structural tag for tool constraints
    ///
    /// Uses structural tags with `triggered_tags` format to force Harmony format output.
    /// This ensures the model outputs in Harmony format (with channels) even when constrained.
    fn generate_harmony_structural_tag(
        tools: &[Tool],
        tool_choice: &Option<ToolChoice>,
    ) -> Result<Option<(String, String)>, Box<Response>> {
        let Some(choice) = tool_choice.as_ref() else {
            return Ok(None);
        };

        match choice {
            ToolChoice::Function { function, .. } => {
                let tag = Self::build_harmony_structural_tag(tools, Some(&function.name))?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            ToolChoice::Value(ToolChoiceValue::Required) => {
                let tag = Self::build_harmony_structural_tag(tools, None)?;
                Ok(Some(("structural_tag".to_string(), tag)))
            }
            ToolChoice::AllowedTools { mode, .. } => {
                if mode == "required" {
                    let tag = Self::build_harmony_structural_tag(tools, None)?;
                    Ok(Some(("structural_tag".to_string(), tag)))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Build Harmony structural tag for tool calling constraints
    fn build_harmony_structural_tag(
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
        if specific_function.is_some() && tools_to_use.is_empty() {
            return Err(Box::new(utils::bad_request_error(format!(
                "Tool '{}' not found in tools list",
                specific_function.unwrap()
            ))));
        }

        // Build tags for each tool
        for tool in tools_to_use {
            let tool_name = &tool.function.name;
            let params_schema = &tool.function.parameters;

            tags.push(json!({
                "begin": format!("<|channel|>commentary to=functions.{}<|constrain|>json<|message|>", tool_name),
                "content": {
                    "type": "json_schema",
                    "json_schema": params_schema
                },
                "end": "" // `end` is empty because <|call|> comes naturally from Harmony stop tokens
            }));
        }

        let stop_after_first = specific_function.is_some();

        let structural_tag = json!({
            "format": {
                "type": "triggered_tags",
                "triggers": ["<|channel|>commentary"],
                "tags": tags,
                "at_least_one": true,
                "stop_after_first": stop_after_first
            }
        });

        serde_json::to_string(&structural_tag).map_err(|e| {
            Box::new(utils::internal_error_message(format!(
                "Failed to serialize structural tag: {}",
                e
            )))
        })
    }
}
