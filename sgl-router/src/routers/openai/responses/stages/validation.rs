//! Validation stage for responses pipeline
//!
//! This stage:
//! - Checks circuit breaker status
//! - Extracts authorization header
//! - Validates basic request parameters

use async_trait::async_trait;
use axum::response::{IntoResponse, Response};

use super::ResponsesStage;
use crate::routers::openai::{responses::ResponsesRequestContext, utils::validate_request};

/// Validation and authentication stage for responses pipeline
pub struct ResponsesValidationStage;

#[async_trait]
impl ResponsesStage for ResponsesValidationStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        // Use shared validation logic
        let validation_output = validate_request(
            &ctx.dependencies.circuit_breaker,
            ctx.input.headers.as_ref(),
            ctx.model(),
        )
        .map_err(|(status, msg)| (status, msg).into_response())?;

        // Store validation output in responses-specific format
        ctx.state.validation = Some(crate::routers::openai::responses::ValidationOutput {
            auth_header: validation_output.auth_header,
            validated_at: validation_output.validated_at,
        });

        Ok(None) // Continue to next stage
    }

    fn name(&self) -> &'static str {
        "ResponsesValidation"
    }
}
