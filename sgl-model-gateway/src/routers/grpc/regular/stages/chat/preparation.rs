//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use std::{borrow::Cow, collections::HashMap};

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    multimodal::{
        types::{ChatContentPart, ImageDetail, Modality, TrackedMedia},
        AsyncMultiModalTracker, TrackerConfig,
    },
    protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{PreparationOutput, RequestContext},
            utils,
        },
    },
};

/// Chat preparation stage
///
/// Processes chat-specific logic including multimodal tracking,
/// tool filtering, and template rendering.
pub(crate) struct ChatPreparationStage;

#[async_trait]
impl PipelineStage for ChatPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.chat_request_arc();
        self.prepare_chat(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatPreparation"
    }
}

impl ChatPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "ChatPreparationStage::prepare_chat").map_err(|e| *e)?;

        // Step 1: Handle Multimodal Tracking for zero-copy extraction
        // We use the tracker to process the messages and extract shared Bytes
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
                                            "Video inputs are not yet supported in gRPC chat",
                                        ));
                                    }
                                };
                                tracker.push_part(chat_part).map_err(|e| {
                                    error!(error = %e, "Failed to push multimodal part to tracker");
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
                error!("Multimodal tracking failed: {}", e);
                error::internal_error("multimodal_tracking_failed", e.to_string())
            })?;

            // Convert TrackedMedia enum (internal) to concrete Arc<ImageFrame> expected by utils
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

        // Step 2: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 3: Process messages and apply chat template
        let processed_messages = match utils::process_chat_messages(
            &body_ref,
            &*tokenizer,
            multimodal_data,
        ) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process chat messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Step 4: Tokenize the processed text (no special tokens - chat template already handles them)
        let encoding = match tokenizer.encode(&processed_messages.text, false) {
            Ok(encoding) => encoding,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Tokenization failed");
                return Err(error::internal_error(
                    "tokenization_failed",
                    format!("Tokenization failed: {}", e),
                ));
            }
        };

        let token_ids = encoding.token_ids().to_vec();

        // Step 5: Build tool constraints if needed
        let tool_call_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, &request.tool_choice, &request.model)
                .map_err(|e| {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request("invalid_tool_configuration", format!("Invalid tool configuration: {}", e))
                })?
        } else {
            None
        };

        // Step 6: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(processed_messages.text.clone()),
            token_ids,
            processed_messages: Some(processed_messages),
            tool_constraints: tool_call_constraint,
            filtered_request: if matches!(body_ref, Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
            // Harmony fields (not used for regular preparation)
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        // Store stop decoder for reuse in response processing
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }
}
