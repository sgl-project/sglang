//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::{error, warn};

use super::PipelineStage;
use crate::{
    core::{ConnectionMode, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
    observability::metrics::{metrics_labels, Metrics},
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::{
        error,
        grpc::context::{RequestContext, WorkerSelection},
    },
};

/// Worker selection stage: Select appropriate worker(s) based on routing mode
pub(crate) struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
}

pub(crate) enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            mode,
        }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        // For Harmony, use selection_text produced during Harmony encoding
        // Otherwise, use original_text from regular preparation
        let text = if prep.harmony_mode {
            prep.selection_text.as_deref()
        } else {
            prep.original_text.as_deref()
        };

        // Get tokens for PrefixHash policy support
        let tokens = if prep.token_ids.is_empty() {
            None
        } else {
            Some(prep.token_ids.as_slice())
        };

        let headers = ctx.input.headers.as_ref();

        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                match self.select_single_worker(
                    ctx.input.model_id.as_deref(),
                    text,
                    tokens,
                    headers,
                ) {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        let model = ctx.input.model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID);
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "Regular",
                            model_id = %model,
                            "No available workers for model"
                        );
                        return Err(error::service_unavailable(
                            "no_available_workers",
                            format!("No available workers for model: {}", model),
                        ));
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode => {
                match self.select_pd_pair(ctx.input.model_id.as_deref(), text, tokens, headers) {
                    Some((prefill, decode)) => WorkerSelection::Dual { prefill, decode },
                    None => {
                        let model = ctx.input.model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID);
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "PrefillDecode",
                            model_id = %model,
                            "No available PD worker pairs for model"
                        );
                        return Err(error::service_unavailable(
                            "no_available_pd_worker_pairs",
                            format!("No available PD worker pairs for model: {}", model),
                        ));
                    }
                }
            }
        };

        ctx.state.workers = Some(workers);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}

impl WorkerSelectionStage {
    fn select_single_worker(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc { port: None }),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        // Use into_iter() to take ownership of Arcs without cloning (avoids atomic inc/dec)
        let available: Vec<Arc<dyn Worker>> =
            workers.into_iter().filter(|w| w.is_available()).collect();

        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(model_id.unwrap_or(UNKNOWN_MODEL_ID));

        // Select worker using the policy
        let idx = policy.select_worker(
            &available,
            &SelectWorkerInfo {
                request_text: text,
                tokens,
                headers,
                hash_ring,
            },
        )?;
        let selected = available[idx].clone();

        // Record worker selection metric
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id.unwrap_or("default"),
            policy.name(),
        );

        Some(selected)
    }

    fn select_pd_pair(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
    ) -> Option<(Arc<dyn Worker>, Arc<dyn Worker>)> {
        let all_workers = self.worker_registry.get_workers_filtered(
            model_id,
            None,
            Some(ConnectionMode::Grpc { port: None }), // Match any gRPC worker
            None,                                      // any runtime type
            false,
        );

        let (available_prefill, available_decode): (Vec<_>, Vec<_>) =
            all_workers
                .into_iter()
                .fold((Vec::new(), Vec::new()), |mut acc, w| {
                    if w.is_available() {
                        match w.metadata().worker_type {
                            WorkerType::Prefill { .. } => acc.0.push(w),
                            WorkerType::Decode => acc.1.push(w),
                            _ => {}
                        }
                    }
                    acc
                });

        if available_prefill.is_empty() {
            warn!("No available prefill workers");
            return None;
        }

        if available_decode.is_empty() {
            warn!("No available decode workers");
            return None;
        }

        // Select using policies
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            hash_ring,
        };
        let prefill_idx = policy.select_worker(&available_prefill, &info)?;
        let decode_idx = policy.select_worker(&available_decode, &info)?;

        let model = model_id.unwrap_or("default");
        let policy_name = policy.name();

        // Record worker selection metrics for both prefill and decode
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_GRPC,
            model,
            policy_name,
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_GRPC,
            model,
            policy_name,
        );

        Some((
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
        ))
    }
}
