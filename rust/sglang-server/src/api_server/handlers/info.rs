//! Model and server metadata: the static cards, and `/server_info` with one
//! scheduler state per DP rank.

use sglang_api_types::api::v1 as api;

use crate::message::ids::Rid;
use crate::message::info::{InternalState, ModelInfo, ServerInfo};
use crate::message::io_struct::{ControlRequest, GetInternalStateReq};

use super::handle::CoreHandle;
use crate::api_server::core::CoreError;

impl CoreHandle {
    /// Return static model metadata without exposing the full launch config.
    pub(crate) fn model_info(&self) -> ModelInfo {
        let metadata = &self.inner.metadata;
        ModelInfo {
            model_path: metadata.model_path.clone(),
            served_model_name: metadata.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            is_generation: true,
            preferred_sampling_params: metadata.preferred_sampling_params.clone(),
            weight_version: metadata.weight_version.clone(),
            load_format: metadata.load_format.clone(),
            reasoning_parser: metadata.reasoning_parser.clone(),
            tool_call_parser: metadata.tool_call_parser.clone(),
            disaggregation_mode: metadata.disaggregation_mode,
        }
    }

    /// Return the served model cards shared by HTTP and gRPC discovery.
    /// This is static metadata; no runtime request is submitted.
    pub(crate) fn list_models(&self) -> Vec<api::ModelCard> {
        let metadata = &self.inner.metadata;
        vec![api::ModelCard {
            id: metadata.served_model_name.clone(),
            root: metadata.served_model_name.clone(),
            parent: None,
            max_model_len: metadata.max_context_length,
        }]
    }

    /// Return public server metadata with the current scheduler metrics.
    /// Raw control bytes and the scheduler's full launch-argument dump never
    /// cross the core boundary.
    pub(crate) async fn server_info(&self) -> Result<ServerInfo, CoreError> {
        let mut internal_states = self.internal_states().await?;
        // Every listener records its own startup timings; the lowest rank's
        // stand for the deployment, as Python's one `TokenizerManager`'s do.
        let startup_time = internal_states
            .iter_mut()
            .find_map(|state| state.rust_server_startup_time.take());
        let metadata = &self.inner.metadata;
        Ok(ServerInfo {
            launch_fields: metadata.launch_server_info.clone(),
            model_path: metadata.model_path.clone(),
            served_model_name: metadata.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            max_context_length: metadata.max_context_length,
            max_total_num_tokens: metadata.max_total_num_tokens,
            version: metadata.version.clone(),
            frontend: "rust",
            startup_time,
            internal_states,
            host: metadata.host.clone(),
            port: metadata.port,
            incremental_streaming_output: metadata.incremental_streaming_output,
            disaggregation_mode: metadata.disaggregation_mode,
            deployment_info: metadata.deployment_info.clone(),
        })
    }

    /// One `InternalState` per DP rank, in rank order, as Python's
    /// `get_internal_state` returns one per DP rank.
    async fn internal_states(&self) -> Result<Vec<InternalState>, CoreError> {
        let envelopes = self
            .control(&|| {
                ControlRequest::GetInternalStateReq(GetInternalStateReq::new(
                    Rid::new().to_string(),
                ))
            })
            .await
            .decode::<InternalStateEnvelope>("internal-state")?;
        Ok(envelopes
            .into_iter()
            .map(|envelope| envelope.internal_state)
            .collect())
    }
}

/// Private scheduler response envelope. This MessagePack shape is decoded at
/// the runtime boundary and deliberately is not part of the core contract.
#[derive(serde::Deserialize)]
struct InternalStateEnvelope {
    #[serde(default)]
    internal_state: InternalState,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::test_utils::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};

    #[test]
    fn list_models_returns_static_cards_without_runtime_work() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: false,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata {
                    served_model_name: "served-model".into(),
                    max_context_length: 1_u64 << 32,
                    ..Default::default()
                },
            },
        );

        assert_eq!(
            handle.list_models(),
            vec![api::ModelCard {
                id: "served-model".into(),
                root: "served-model".into(),
                parent: None,
                max_model_len: 1_u64 << 32,
            }]
        );
        assert!(intake_rx.try_recv().is_err());
        assert!(abort_rx.try_recv().is_err());
    }

    /// `/server_info` lists one internal state per DP rank, in rank order:
    /// a target with a lower rank than this listener's comes first.
    #[tokio::test]
    async fn server_info_lists_internal_states_in_rank_order() {
        use crate::api_server::core::fanout::testing::DummyTarget;
        let envelope = |throughput: f64| {
            bytes::Bytes::from(
                rmp_serde::to_vec_named(&serde_json::json!({
                    "internal_state": {"last_gen_throughput": throughput},
                }))
                .unwrap(),
            )
        };
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                metadata: CoreMetadata {
                    deployment_info: crate::message::config::DeploymentInfo {
                        dp_rank: Some(1),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: vec![DummyTarget::replying(&[0], move |_| Ok(envelope(2.5)))],
            },
        );

        let (info, ()) = tokio::join!(
            handle.server_info(),
            answer_local_control(&intake_rx, envelope(1.5)),
        );
        let throughputs: Vec<_> = info
            .unwrap()
            .internal_states
            .iter()
            .map(|state| state.last_gen_throughput)
            .collect();
        assert_eq!(throughputs, [Some(2.5), Some(1.5)]);
    }
}
