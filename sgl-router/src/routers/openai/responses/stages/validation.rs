//! Validation stage for responses pipeline
//!
//! This stage:
//! - Checks circuit breaker status
//! - Extracts authorization header
//! - Validates basic request parameters

use std::time::Instant;

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::ResponsesStage;
use crate::routers::openai::{responses::ResponsesRequestContext, utils::extract_auth_header};

/// Validation and authentication stage for responses pipeline
pub struct ResponsesValidationStage;

#[async_trait]
impl ResponsesStage for ResponsesValidationStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        // 1. Circuit breaker check
        if !ctx.dependencies.circuit_breaker.can_execute() {
            return Err((StatusCode::SERVICE_UNAVAILABLE, "Circuit breaker open").into_response());
        }

        // 2. Extract authorization header
        let auth_header = extract_auth_header(ctx.input.headers.as_ref()).map(|s| s.to_string());

        // 3. Validate model is specified
        let model = ctx.model();
        if model.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "Model parameter is required and cannot be empty",
            )
                .into_response());
        }

        // 4. Store validation output
        ctx.state.validation = Some(crate::routers::openai::responses::ValidationOutput {
            auth_header,
            validated_at: Instant::now(),
        });

        Ok(None) // Continue to next stage
    }

    fn name(&self) -> &'static str {
        "ResponsesValidation"
    }
}
