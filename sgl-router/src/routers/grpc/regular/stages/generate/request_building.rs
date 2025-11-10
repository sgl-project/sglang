//! Generate request building stage: Build proto GenerateRequest for generate requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::grpc::{
    common::stages::{helpers, PipelineStage},
    context::{ClientSelection, RequestContext, WorkerSelection},
    error,
};

/// Generate request building stage
///
/// Extracts generate-specific request building logic from the old unified RequestBuildingStage.
pub struct GenerateRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl GenerateRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for GenerateRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error("Client acquisition not completed")
        })?;

        let generate_request = ctx.generate_request_arc();

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Build generate request
        let request_id = generate_request
            .rid
            .clone()
            .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

        let mut proto_request = builder_client
            .build_plain_generate_request(
                request_id,
                &generate_request,
                prep.original_text.clone(),
                prep.token_ids.clone(),
            )
            .map_err(|e| {
                error!(function = "GenerateRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request(e)
            })?;

        // Inject PD metadata if needed
        if self.inject_pd_metadata {
            if let WorkerSelection::Dual { prefill, .. } = ctx.state.workers.as_ref().unwrap() {
                helpers::inject_bootstrap_metadata(&mut proto_request, prefill);
            }
        }

        ctx.state.proto_request = Some(proto_request);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "GenerateRequestBuilding"
    }
}
