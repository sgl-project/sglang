//! Model and server metadata: the static cards, and `/server_info` with one
//! scheduler state per DP rank.

use sglang_api_types::api::v1 as api;

use crate::message::ids::Rid;
use crate::message::info::{DpListener, InternalState, ModelInfo, ServerInfo};
use crate::message::io_struct::{ControlRequest, GetHiCacheL3LayoutReq, GetInternalStateReq};

use std::sync::Arc;

use super::frontend_state::ModelFacts;
use super::handle::CoreHandle;
use crate::api_server::core::CoreError;

impl CoreHandle {
    /// The scheduler-owned model facts: the cache, refreshed from the local
    /// scheduler when a control call since may have changed them. A failed
    /// refresh keeps the last facts (logged), so metadata reads never fail
    /// because a scheduler is busy.
    pub(crate) async fn model_facts(&self) -> Arc<ModelFacts> {
        let frontend = &self.inner.frontend;
        if !frontend.is_stale() {
            return frontend.cached().await;
        }
        match self.local_model_facts().await {
            Ok(facts) => frontend.store(facts, true).await,
            Err(error) => {
                tracing::warn!(%error, "model facts refresh failed; serving the last known");
                frontend.cached().await
            }
        }
    }

    /// The served model name a request's `model` must match.
    pub(crate) async fn served_model_name(&self) -> String {
        self.model_facts().await.served_model_name.clone()
    }

    async fn local_model_facts(&self) -> Result<ModelFacts, CoreError> {
        let bytes = self
            .submit_control_req(ControlRequest::GetInternalStateReq(
                GetInternalStateReq::new(Rid::new().to_string()),
            ))
            .await?;
        rmp_serde::from_slice::<ModelFactsEnvelope>(&bytes)
            .map(|envelope| envelope.internal_state)
            .map_err(|error| {
                CoreError::InvalidResponse(format!("invalid internal-state response: {error}"))
            })
    }

    /// Return model metadata without exposing the full launch config.
    pub(crate) async fn model_info(&self) -> ModelInfo {
        let facts = self.model_facts().await;
        let metadata = &self.inner.metadata;
        ModelInfo {
            model_path: facts.model_path.clone(),
            served_model_name: facts.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            is_generation: true,
            preferred_sampling_params: metadata.preferred_sampling_params.clone(),
            weight_version: facts.weight_version.clone(),
            load_format: facts.load_format.clone(),
            reasoning_parser: metadata.reasoning_parser.clone(),
            tool_call_parser: metadata.tool_call_parser.clone(),
            disaggregation_mode: facts.disaggregation_mode,
            model_type: metadata.model_type.clone(),
            architectures: metadata.architectures.clone(),
            launch_fields: metadata.launch_model_info.clone(),
        }
    }

    /// Return the served model cards shared by HTTP and gRPC discovery.
    pub(crate) async fn list_models(&self) -> Vec<api::ModelCard> {
        let facts = self.model_facts().await;
        vec![api::ModelCard {
            id: facts.served_model_name.clone(),
            root: facts.served_model_name.clone(),
            parent: None,
            max_model_len: self.inner.metadata.max_context_length,
        }]
    }

    /// Return public server metadata with the current scheduler metrics.
    /// Raw control bytes and the scheduler's full launch-argument dump never
    /// cross the core boundary.
    pub(crate) async fn server_info(&self) -> Result<ServerInfo, CoreError> {
        let mut internal_states = self.internal_states().await?;
        let facts = self.model_facts().await;
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
            model_path: facts.model_path.clone(),
            served_model_name: facts.served_model_name.clone(),
            tokenizer_path: metadata.tokenizer_path.clone(),
            max_context_length: metadata.max_context_length,
            max_total_num_tokens: metadata.max_total_num_tokens,
            num_reserved_tokens: metadata.num_reserved_tokens,
            version: metadata.version.clone(),
            frontend: "rust",
            startup_time,
            internal_states,
            host: metadata.host.clone(),
            port: metadata.port,
            incremental_streaming_output: metadata.incremental_streaming_output,
            disaggregation_mode: facts.disaggregation_mode,
            allow_auto_truncate: metadata.allow_auto_truncate,
            skip_tokenizer_init: metadata.skip_tokenizer_init,
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

/// The model facts inside the scheduler's internal-state envelope, which
/// starts from the resolved server args (so a weight update's new
/// `model_path` / `weight_version` and a role switch's mode are in it).
#[derive(serde::Deserialize)]
struct ModelFactsEnvelope {
    internal_state: ModelFacts,
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

    #[tokio::test]
    async fn list_models_returns_static_cards_without_runtime_work() {
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
            handle.list_models().await,
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

/// Python's `disabled_l3_layout(reason)`.
fn disabled_l3_layout(reason: &str) -> serde_json::Value {
    serde_json::json!({
        "enabled": false,
        "backend": null,
        "page_size": null,
        "hicache_object_layout": {"pools": []},
        "reason": reason,
    })
}

fn l3_layout_pp_rank(layout: &serde_json::Value) -> Option<i64> {
    layout["hicache_object_layout"]
        .as_object()
        .and_then(|inner| inner.get("pp_rank"))
        .and_then(serde_json::Value::as_i64)
}

/// The part of a reply that must agree across pipeline stages.
fn l3_layout_stage_independent(layout: &serde_json::Value) -> serde_json::Value {
    let Some(inner) = layout["hicache_object_layout"].as_object() else {
        return layout.clone();
    };
    let rest: serde_json::Map<String, serde_json::Value> = inner
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "pp_rank" | "pools"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let mut layout = layout.clone();
    layout["hicache_object_layout"] = serde_json::Value::Object(rest);
    layout
}

fn l3_layout_unavailable(layout: &serde_json::Value, reason: &str) -> serde_json::Value {
    let mut layout = layout.clone();
    layout["enabled"] = serde_json::Value::Bool(false);
    layout["hicache_object_layout"]["pools"] = serde_json::json!([]);
    layout["reason"] = serde_json::Value::String(reason.into());
    layout
}

/// Python's `merge_l3_layouts`: replicas must match, pipeline stages are
/// concatenated. `None` when every rank sent none (an older engine).
pub(crate) fn merge_l3_layouts(layouts: &[Option<serde_json::Value>]) -> Option<serde_json::Value> {
    if layouts.iter().all(Option::is_none) {
        return None;
    }
    let inconsistent = "Inconsistent HiCache L3 layouts across schedulers";
    let mut stages: std::collections::BTreeMap<Option<i64>, &serde_json::Value> =
        std::collections::BTreeMap::new();
    for layout in layouts {
        let Some(layout) = layout else {
            return Some(disabled_l3_layout(inconsistent));
        };
        if *stages.entry(l3_layout_pp_rank(layout)).or_insert(layout) != layout {
            return Some(disabled_l3_layout(inconsistent));
        }
    }
    if stages.contains_key(&None) {
        return Some(if stages.len() == 1 {
            layouts[0].clone().expect("checked above")
        } else {
            disabled_l3_layout(inconsistent)
        });
    }
    let ordered: Vec<&serde_json::Value> = stages.values().copied().collect();
    let first = ordered[0];
    let first_independent = l3_layout_stage_independent(first);
    if ordered[1..]
        .iter()
        .any(|stage| l3_layout_stage_independent(stage) != first_independent)
    {
        return Some(disabled_l3_layout(inconsistent));
    }
    if first["enabled"].as_bool() != Some(true) {
        return Some(first.clone());
    }
    let inner = &first["hicache_object_layout"];
    let pp_size = match inner.get("pp_size") {
        None => 1,
        Some(value) => match value.as_i64() {
            Some(size) if !value.is_f64() => size,
            _ => return Some(disabled_l3_layout(inconsistent)),
        },
    };
    let ranks: Vec<i64> = stages.keys().flatten().copied().collect();
    if ranks != (0..pp_size).collect::<Vec<_>>() {
        return Some(l3_layout_unavailable(
            first,
            "Incomplete pipeline-parallel stage layouts",
        ));
    }
    if pp_size == 1 {
        return Some(first.clone());
    }
    // A layout spanning several stages belongs to none of them: pp_rank is null.
    let pools: Vec<serde_json::Value> = ordered
        .iter()
        .flat_map(|stage| {
            stage["hicache_object_layout"]["pools"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .collect();
    let mut merged = first.clone();
    merged["hicache_object_layout"]["pools"] = serde_json::Value::Array(pools);
    merged["hicache_object_layout"]["pp_rank"] = serde_json::Value::Null;
    Some(merged)
}

/// The scheduler's `GetHiCacheL3LayoutReqOutput`.
#[derive(Debug, serde::Deserialize)]
struct L3LayoutOutput {
    #[serde(default)]
    layout: Option<serde_json::Value>,
}

/// Python's `get_elastic_ep_state`, as the Rust frontend reports it: the
/// launch's EP size with no scale pending, since it serves no scale.
#[derive(Debug, PartialEq)]
pub(crate) struct ElasticEpState {
    pub(crate) effective_ep_size: i64,
}

impl CoreHandle {
    /// `/get_hicache_l3_cache_layout`: every DP rank's L3 layout, merged as
    /// Python merges them; the disabled envelope when no rank sent one.
    pub(crate) async fn hicache_l3_cache_layout(&self) -> Result<serde_json::Value, CoreError> {
        let outputs = self
            .control(&|| {
                ControlRequest::GetHiCacheL3LayoutReq(GetHiCacheL3LayoutReq::new(
                    Rid::new().to_string(),
                ))
            })
            .await
            .decode::<L3LayoutOutput>("get-hicache-l3-cache-layout")?;
        let layouts: Vec<Option<serde_json::Value>> =
            outputs.into_iter().map(|output| output.layout).collect();
        Ok(merge_l3_layouts(&layouts)
            .unwrap_or_else(|| disabled_l3_layout("No scheduler layout replies")))
    }

    /// The elastic-EP state, or `None` without `--elastic-ep-backend`.
    pub(crate) fn elastic_ep_state(&self) -> Option<ElasticEpState> {
        let info = &self.inner.metadata.deployment_info;
        info.elastic_ep_backend.as_ref()?;
        Some(ElasticEpState {
            effective_ep_size: i64::from(info.dp_size.max(1)),
        })
    }
}

#[cfg(test)]
mod l3_and_elastic_tests {
    use super::*;

    /// Python's `merge_l3_layouts`: every reply absent is `None`; a mix is
    /// the disabled envelope; replicas must match; pipeline stages
    /// concatenate their pools; a missing stage is unavailable.
    #[test]
    fn l3_layouts_merge_as_python_merges_them() {
        let stage = |rank: i64, pools: serde_json::Value| {
            serde_json::json!({
                "enabled": true, "backend": "s3", "page_size": 64,
                "hicache_object_layout": {"pp_size": 2, "pp_rank": rank, "pools": pools}
            })
        };
        assert_eq!(merge_l3_layouts(&[None, None]), None);
        assert_eq!(merge_l3_layouts(&[]), None);
        let mixed = merge_l3_layouts(&[Some(stage(0, serde_json::json!([]))), None]).unwrap();
        assert_eq!(mixed["enabled"], false);
        assert_eq!(
            mixed["reason"],
            "Inconsistent HiCache L3 layouts across schedulers"
        );

        let single = serde_json::json!({
            "enabled": true, "hicache_object_layout": {"pools": [1]}
        });
        assert_eq!(
            merge_l3_layouts(&[Some(single.clone()), Some(single.clone())]),
            Some(single.clone())
        );
        let other = serde_json::json!({"enabled": true, "hicache_object_layout": {"pools": [2]}});
        assert_eq!(
            merge_l3_layouts(&[Some(single), Some(other)]).unwrap()["enabled"],
            false,
            "replicas must agree"
        );

        let merged = merge_l3_layouts(&[
            Some(stage(1, serde_json::json!(["b"]))),
            Some(stage(0, serde_json::json!(["a"]))),
        ])
        .unwrap();
        assert_eq!(
            merged["hicache_object_layout"]["pools"],
            serde_json::json!(["a", "b"])
        );
        assert!(merged["hicache_object_layout"]["pp_rank"].is_null());
        assert_eq!(merged["backend"], "s3");

        let partial = merge_l3_layouts(&[Some(stage(0, serde_json::json!(["a"])))]).unwrap();
        assert_eq!(partial["enabled"], false);
        assert_eq!(
            partial["reason"],
            "Incomplete pipeline-parallel stage layouts"
        );
        assert_eq!(
            partial["hicache_object_layout"]["pools"],
            serde_json::json!([])
        );

        let disabled = serde_json::json!({
            "enabled": false, "hicache_object_layout": {"pp_size": 1, "pp_rank": 0, "pools": []},
            "reason": "off"
        });
        assert_eq!(merge_l3_layouts(&[Some(disabled.clone())]), Some(disabled));
    }

    /// The layout read fans out and merges the ranks; the elastic state is
    /// absent without a backend and idle with one.
    #[tokio::test]
    async fn l3_layout_is_read_from_every_rank_and_elastic_state_follows_the_launch() {
        use crate::api_server::handlers::test_utils::*;
        use crate::api_server::handlers::{CoreConfig, CoreMetadata};
        use crate::message::config::DeploymentInfo;
        let harness = Harness::unbounded(8);
        let layout = serde_json::json!({"enabled": false, "hicache_object_layout": {"pools": []}});
        let handle = harness.handle.clone();
        let (merged, ()) = tokio::join!(
            handle.hicache_l3_cache_layout(),
            answer_local_control(
                &harness.intake_rx,
                bytes::Bytes::from(
                    rmp_serde::to_vec_named(&serde_json::json!({"layout": layout})).unwrap()
                ),
            ),
        );
        assert_eq!(merged.unwrap(), layout);
        assert!(harness.handle.elastic_ep_state().is_none());

        let (intake_tx, _intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let elastic = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                multimodal: false,
                metrics: None,
                request_log: None,
                generate_validator: None,
                mm_limits: Vec::new(),
                metadata: CoreMetadata {
                    deployment_info: DeploymentInfo {
                        dp_size: 4,
                        elastic_ep_backend: Some("mooncake".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: Vec::new(),
            },
        );
        assert_eq!(
            elastic.elastic_ep_state(),
            Some(ElasticEpState {
                effective_ep_size: 4
            })
        );
    }
}
