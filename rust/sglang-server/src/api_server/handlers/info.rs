//! Model and server metadata: the static cards, and `/server_info` with one
//! scheduler state per DP rank.

use sglang_api_types::api::v1 as api;

use crate::message::ids::Rid;
use crate::message::info::{DpListener, InternalState, ModelInfo, ServerInfo};
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
            weight_version: metadata.weight_version.as_deref().map(str::to_owned),
            load_format: metadata.load_format.clone(),
            reasoning_parser: metadata.reasoning_parser.clone(),
            tool_call_parser: metadata.tool_call_parser.clone(),
            disaggregation_mode: metadata.disaggregation_mode,
            launch_fields: metadata.launch_model_info.clone(),
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
        // The listener serving the deployment's base URL, DP rank 0's,
        // publishes every rank's listener (`rust_server_endpoints`), by rank.
        let mut dp_listeners: Vec<DpListener> = if self.inner.dp_rank == 0 {
            metadata
                .deployment_info
                .rust_server_endpoints
                .iter()
                .map(|endpoint| DpListener {
                    dp_rank: endpoint.dp_rank,
                    url: endpoint.endpoint.clone(),
                })
                .collect()
        } else {
            Vec::new()
        };
        dp_listeners.sort_by_key(|listener| listener.dp_rank);
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
            dp_listeners,
        })
    }

    /// One `InternalState` per DP rank, in rank order, as Python's
    /// `get_internal_state` returns one per DP rank.
    async fn internal_states(&self) -> Result<Vec<InternalState>, CoreError> {
        self.control(&|| {
            ControlRequest::GetInternalStateReq(GetInternalStateReq::new(Rid::new().to_string()))
        })
        .await
        .decode_with("internal-state", |envelope: InternalStateEnvelope| {
            InternalState::from_scheduler(
                envelope.internal_state,
                &self.inner.metadata.server_info_state_keys,
            )
            .map_err(|error| error.to_string())
        })
    }
}

/// Private scheduler response envelope. This MessagePack shape is decoded at
/// the runtime boundary and deliberately is not part of the core contract;
/// the raw state never leaves [`CoreHandle::internal_states`].
#[derive(serde::Deserialize)]
struct InternalStateEnvelope {
    #[serde(default = "empty_state")]
    internal_state: rmpv::Value,
}

fn empty_state() -> rmpv::Value {
    rmpv::Value::Map(Vec::new())
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
                multimodal: false,
                metrics: None,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata {
                    served_model_name: "served-model".into(),
                    max_context_length: 1_u64 << 32,
                    ..Default::default()
                },
                request_log: None,
                generate_validator: None,
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
    /// a target with a lower rank than this listener's comes first. A rank
    /// whose state breaks the typed contract fails it, named in the error.
    #[tokio::test]
    async fn server_info_lists_internal_states_in_rank_order() {
        let throughputs: Vec<_> = rank_one_server_info(serde_json::json!(2.5))
            .await
            .unwrap()
            .internal_states
            .iter()
            .map(|state| state.last_gen_throughput)
            .collect();
        assert_eq!(throughputs, [Some(2.5), Some(1.5)]);

        let error = rank_one_server_info(serde_json::json!("fast"))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("dp rank 0: invalid internal-state response"),
            "{error}"
        );
    }

    /// `server_info` on the listener of DP rank 1, whose state reports a
    /// throughput of 1.5, while its peer, DP rank 0, reports `peer_throughput`.
    async fn rank_one_server_info(
        peer_throughput: serde_json::Value,
    ) -> Result<ServerInfo, CoreError> {
        use crate::api_server::core::fanout::testing::DummyTarget;
        let envelope = |throughput: serde_json::Value| {
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
                multimodal: false,
                metrics: None,
                metadata: CoreMetadata {
                    deployment_info: crate::message::config::DeploymentInfo {
                        dp_rank: Some(1),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: vec![DummyTarget::replying(&[0], move |_| {
                    Ok(envelope(peer_throughput.clone()))
                })],
                request_log: None,
                generate_validator: None,
            },
        );

        let (info, ()) = tokio::join!(
            handle.server_info(),
            answer_local_control(&intake_rx, envelope(serde_json::json!(1.5))),
        );
        info
    }
}
