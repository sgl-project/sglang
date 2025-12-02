//! Validation stage for chat pipeline
//!
//! This stage:
//! - Checks circuit breaker status
//! - Extracts authorization header
//! - Validates basic request parameters

use async_trait::async_trait;
use axum::response::{IntoResponse, Response};

use super::ChatStage;
use crate::routers::openai::{chat::ChatRequestContext, utils::validate_request};

/// Validation and authentication stage for chat pipeline
pub struct ChatValidationStage;

#[async_trait]
impl ChatStage for ChatValidationStage {
    async fn execute(&self, ctx: &mut ChatRequestContext) -> Result<(), Response> {
        // Use shared validation logic
        let validation_output = validate_request(
            &ctx.dependencies.circuit_breaker,
            ctx.input.headers.as_ref(),
            ctx.model(),
        )
        .map_err(|(status, msg)| (status, msg).into_response())?;

        // Store validation output in chat-specific format
        ctx.state.validation = Some(crate::routers::openai::chat::ValidationOutput {
            auth_header: validation_output.auth_header,
            validated_at: validation_output.validated_at,
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChatValidation"
    }
}
