//! Weight management: process-group setup, remote-instance sends, parameter
//! reads, and the weight updates. An update holds the frontend's update lock
//! on every listener while it runs (`FrontendState::hold_for_update`): the
//! origin listener holds its own, and the peer relay holds each other
//! listener's around its scheduler's copy.

use crate::message::ids::Rid;
use crate::message::io_struct::{
    BeginWeightUpdateReqInput, ControlRequest, DestroyWeightsUpdateGroupReqInput, DiskWeightUpdate,
    DistributedWeightUpdate, EndWeightUpdateReqInput, GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput, InitWeightsUpdateGroupReqInput, IpcWeightUpdate,
    PluginWeightUpdate, PluginWeightUpdateReqInput, SendWeightsToRemoteInstanceReqInput,
    TensorWeightUpdate, UpdateWeightFromDiskReqInput, UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput, UpdateWeightsFromTensorReqInput, WeightsSendGroup,
    WeightsUpdateGroup,
};

use super::generate::AbortEnding;

use super::handle::CoreHandle;
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout::{self, RequestBuilder};

impl CoreHandle {
    /// Read `name` from every DP rank, each dimension truncated to
    /// `truncate_size` elements: one JSON value per rank in rank order, `null`
    /// where a rank found no such parameter.
    pub(crate) async fn get_weights_by_name(
        &self,
        name: String,
        truncate_size: i64,
    ) -> Result<Vec<serde_json::Value>, CoreError> {
        if name.is_empty() {
            return Err(CoreError::InvalidArgument("missing field `name`".into()));
        }
        let outputs = self
            .control(&|| {
                ControlRequest::GetWeightsByNameReqInput(GetWeightsByNameReqInput::new(
                    Rid::new().to_string(),
                    name.clone(),
                    truncate_size,
                ))
            })
            .await
            .decode::<WeightsByNameOutput>("get-weights-by-name")?;
        Ok(outputs
            .into_iter()
            .map(|output| output.parameter.unwrap_or(serde_json::Value::Null))
            .collect())
    }

    pub(crate) async fn init_weights_update_group(
        &self,
        group: WeightsUpdateGroup,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("init_weights_update_group", &|| {
            ControlRequest::InitWeightsUpdateGroupReqInput(InitWeightsUpdateGroupReqInput::new(
                Rid::new().to_string(),
                group.clone(),
            ))
        })
        .await
    }

    pub(crate) async fn destroy_weights_update_group(
        &self,
        group_name: String,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("destroy_weights_update_group", &|| {
            ControlRequest::DestroyWeightsUpdateGroupReqInput(
                DestroyWeightsUpdateGroupReqInput::new(Rid::new().to_string(), group_name.clone()),
            )
        })
        .await
    }

    pub(crate) async fn init_weights_send_group_for_remote_instance(
        &self,
        group: WeightsSendGroup,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("init_weights_send_group_for_remote_instance", &|| {
            ControlRequest::InitWeightsSendGroupForRemoteInstanceReqInput(
                InitWeightsSendGroupForRemoteInstanceReqInput::new(
                    Rid::new().to_string(),
                    group.clone(),
                ),
            )
        })
        .await
    }

    pub(crate) async fn send_weights_to_remote_instance(
        &self,
        master_address: String,
        ports: String,
        group_name: String,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("send_weights_to_remote_instance", &|| {
            ControlRequest::SendWeightsToRemoteInstanceReqInput(
                SendWeightsToRemoteInstanceReqInput::new(
                    Rid::new().to_string(),
                    master_address.clone(),
                    ports.clone(),
                    group_name.clone(),
                ),
            )
        })
        .await
    }

    /// The group operations Python restricts to one DP rank (its
    /// `assert dp_size == 1`); kept as a 400 here rather than lifted, so the
    /// first pass behaves as the Python server does on the deployments it
    /// supports. Merged as `merge_results`.
    async fn single_rank_group(
        &self,
        what: &str,
        build: &RequestBuilder<'_>,
    ) -> Result<WeightGroupOutcome, CoreError> {
        let dp_size = self.inner.metadata.deployment_info.dp_size.max(1);
        if dp_size > 1 {
            return Err(CoreError::InvalidArgument(format!(
                "data-parallel replicas are not supported for {what} (dp_size={dp_size})"
            )));
        }
        let outcomes = self
            .control(build)
            .await
            .decode::<WeightGroupOutcome>(what)?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        Ok(WeightGroupOutcome { success, message })
    }
}

/// Python's `Literal["target", "draft", "all"]`; anything else is the 422
/// Python's request type raises, a 400 here.
fn check_selector(selector: &str) -> Result<(), CoreError> {
    match selector {
        "target" | "draft" | "all" => Ok(()),
        other => Err(CoreError::InvalidArgument(format!(
            "selector: Input should be 'target', 'draft' or 'all', got {other:?}"
        ))),
    }
}

impl CoreHandle {
    /// Run one weight update under this listener's update hold, as Python's
    /// `_weight_update_session_replies`: aborts first when asked (Python's
    /// `abort_request(abort_all=True)`), then nothing is admitted here until
    /// the scheduler ranks have answered.
    async fn weight_update<T: serde::de::DeserializeOwned>(
        &self,
        what: &str,
        abort_all_requests: bool,
        build: &RequestBuilder<'_>,
    ) -> Result<Vec<T>, CoreError> {
        if abort_all_requests {
            self.abort(None, AbortEnding::AbortFinishReason);
        }
        let hold = self.inner.frontend.hold_for_update().await;
        let outcomes = self.control(build).await.decode::<T>(what);
        drop(hold);
        outcomes
    }

    /// Reload the weights from disk on every DP rank. `load_format` falls
    /// back to the launch's, as Python's TokenizerManager fills it; the
    /// outcome is merged like Python's `_wait_for_model_update_from_disk`,
    /// with one `num_paused_requests` per rank.
    pub(crate) async fn update_weights_from_disk(
        &self,
        mut update: DiskWeightUpdate,
    ) -> Result<DiskUpdateOutcome, CoreError> {
        if update.model_path.is_empty() {
            return Err(CoreError::InvalidArgument(
                "missing field `model_path`".into(),
            ));
        }
        if update.load_format.is_none() {
            update.load_format = self.inner.metadata.load_format.clone();
        }
        let abort_all = update.abort_all_requests;
        let outcomes: Vec<DiskUpdateOutput> = self
            .weight_update("update-weights-from-disk", abort_all, &|| {
                ControlRequest::UpdateWeightFromDiskReqInput(UpdateWeightFromDiskReqInput::new(
                    Rid::new().to_string(),
                    update.clone(),
                ))
            })
            .await?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        Ok(DiskUpdateOutcome {
            success,
            message: weight_version_suffix(message, success, update.weight_version.as_deref()),
            num_paused_requests: outcomes
                .iter()
                .map(|outcome| outcome.num_paused_requests)
                .collect(),
        })
    }

    pub(crate) async fn update_weights_from_distributed(
        &self,
        update: DistributedWeightUpdate,
    ) -> Result<WeightUpdateOutcome, CoreError> {
        check_selector(&update.selector)?;
        self.single_rank_update(
            "update_weights_from_distributed",
            update.abort_all_requests,
            update.weight_version.clone(),
            &|| {
                ControlRequest::UpdateWeightsFromDistributedReqInput(
                    UpdateWeightsFromDistributedReqInput::new(
                        Rid::new().to_string(),
                        update.clone(),
                    ),
                )
            },
        )
        .await
    }

    pub(crate) async fn update_weights_from_tensor(
        &self,
        update: TensorWeightUpdate,
    ) -> Result<WeightUpdateOutcome, CoreError> {
        check_selector(&update.selector)?;
        self.single_rank_update(
            "update_weights_from_tensor",
            update.abort_all_requests,
            update.weight_version.clone(),
            &|| {
                ControlRequest::UpdateWeightsFromTensorReqInput(
                    UpdateWeightsFromTensorReqInput::new(Rid::new().to_string(), update.clone()),
                )
            },
        )
        .await
    }

    /// Python wraps this one's single-rank check in its failure message
    /// rather than raising, so a multi-rank deployment answers
    /// `{success: false}` here too.
    pub(crate) async fn update_weights_from_ipc(
        &self,
        update: IpcWeightUpdate,
    ) -> Result<WeightUpdateOutcome, CoreError> {
        if update.zmq_handles.is_empty() {
            return Err(CoreError::InvalidArgument(
                "missing field `zmq_handles`".into(),
            ));
        }
        let dp_size = self.inner.metadata.deployment_info.dp_size.max(1);
        if dp_size > 1 {
            return Ok(WeightUpdateOutcome {
                success: false,
                message: format!(
                    "IPC weight update failed: data-parallel replicas are not supported for \
                     update_weights_from_ipc (dp_size={dp_size})"
                ),
            });
        }
        self.single_rank_update(
            "update_weights_from_ipc",
            false,
            update.weight_version.clone(),
            &|| {
                ControlRequest::UpdateWeightsFromIPCReqInput(UpdateWeightsFromIPCReqInput::new(
                    Rid::new().to_string(),
                    update.clone(),
                ))
            },
        )
        .await
    }

    /// Open a weight-update session on every DP rank's selected runners.
    pub(crate) async fn begin_weight_update(
        &self,
        selector: String,
    ) -> Result<WeightUpdateOutcome, CoreError> {
        check_selector(&selector)?;
        let outcomes: Vec<WeightUpdateOutcome> = self
            .weight_update("begin-weight-update", false, &|| {
                ControlRequest::BeginWeightUpdateReqInput(BeginWeightUpdateReqInput::new(
                    Rid::new().to_string(),
                    selector.clone(),
                ))
            })
            .await?;
        Ok(merge_update(&outcomes))
    }

    /// Close the session opened by [`CoreHandle::begin_weight_update`].
    pub(crate) async fn end_weight_update(&self) -> Result<WeightUpdateOutcome, CoreError> {
        let outcomes: Vec<WeightUpdateOutcome> = self
            .weight_update("end-weight-update", false, &|| {
                ControlRequest::EndWeightUpdateReqInput(EndWeightUpdateReqInput::new(
                    Rid::new().to_string(),
                ))
            })
            .await?;
        Ok(merge_update(&outcomes))
    }

    /// Run the platform's named weight-update handler; merged as Python's
    /// `PluginWeightUpdateReqOutput.merged`.
    pub(crate) async fn plugin_weight_update(
        &self,
        update: PluginWeightUpdate,
    ) -> Result<PluginUpdateOutcome, CoreError> {
        if update.handler.is_empty() {
            return Err(CoreError::InvalidArgument("missing field `handler`".into()));
        }
        check_selector(&update.selector)?;
        self.require_single_rank("plugin_weight_update")?;
        let outcomes: Vec<PluginUpdateOutput> = self
            .weight_update("plugin-weight-update", update.abort_all_requests, &|| {
                ControlRequest::PluginWeightUpdateReqInput(PluginWeightUpdateReqInput::new(
                    Rid::new().to_string(),
                    update.clone(),
                ))
            })
            .await?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        Ok(PluginUpdateOutcome {
            success,
            message,
            received_tensor_counts: match (success, outcomes.first()) {
                (true, Some(first)) => first.received_tensor_counts.clone(),
                _ => Vec::new(),
            },
        })
    }

    /// The updates Python restricts to one DP rank (its `assert dp_size == 1`),
    /// merged as `merge_results` with the weight-version note appended.
    async fn single_rank_update(
        &self,
        what: &str,
        abort_all_requests: bool,
        weight_version: Option<String>,
        build: &RequestBuilder<'_>,
    ) -> Result<WeightUpdateOutcome, CoreError> {
        self.require_single_rank(what)?;
        let outcomes: Vec<WeightUpdateOutcome> =
            self.weight_update(what, abort_all_requests, build).await?;
        let merged = merge_update(&outcomes);
        Ok(WeightUpdateOutcome {
            message: weight_version_suffix(
                merged.message,
                merged.success,
                weight_version.as_deref(),
            ),
            success: merged.success,
        })
    }

    fn require_single_rank(&self, what: &str) -> Result<(), CoreError> {
        let dp_size = self.inner.metadata.deployment_info.dp_size.max(1);
        if dp_size > 1 {
            return Err(CoreError::InvalidArgument(format!(
                "data-parallel replicas are not supported for {what} (dp_size={dp_size})"
            )));
        }
        Ok(())
    }
}

/// Python's `_update_weight_version_if_provided` note on a successful update
/// that named a version. The version itself is recorded by the scheduler
/// and read back with the model facts.
fn weight_version_suffix(message: String, success: bool, weight_version: Option<&str>) -> String {
    match weight_version {
        Some(version) if success => format!("{message} Weight version updated to {version}."),
        _ => message,
    }
}

/// `FanOutCommunicator.merge_results`: every rank succeeded, messages joined.
fn merge_update(outcomes: &[WeightUpdateOutcome]) -> WeightUpdateOutcome {
    let (success, message) = fanout::merge_status(
        outcomes
            .iter()
            .map(|outcome| (outcome.success, outcome.message.as_str())),
    );
    WeightUpdateOutcome { success, message }
}

/// The scheduler's `UpdateWeightFromDiskReqOutput`.
#[derive(Debug, serde::Deserialize)]
struct DiskUpdateOutput {
    success: bool,
    #[serde(default)]
    message: String,
    #[serde(default)]
    num_paused_requests: i64,
}

/// `/update_weights_from_disk`'s merged outcome: Python's
/// `{success, message, num_paused_requests}`, the count per DP rank.
#[derive(Debug, PartialEq)]
pub(crate) struct DiskUpdateOutcome {
    pub(crate) success: bool,
    pub(crate) message: String,
    pub(crate) num_paused_requests: Vec<i64>,
}

/// The `{success, message}` output of the other weight updates and the
/// session calls, on the wire and merged.
#[derive(Debug, Clone, serde::Deserialize, PartialEq)]
pub(crate) struct WeightUpdateOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

/// The scheduler's `PluginWeightUpdateReqOutput`.
#[derive(Debug, serde::Deserialize)]
struct PluginUpdateOutput {
    success: bool,
    #[serde(default)]
    message: String,
    #[serde(default)]
    received_tensor_counts: Vec<i64>,
}

/// Python's `PluginWeightUpdateReqOutput.merged`.
#[derive(Debug, PartialEq)]
pub(crate) struct PluginUpdateOutcome {
    pub(crate) success: bool,
    pub(crate) message: String,
    pub(crate) received_tensor_counts: Vec<i64>,
}

/// Python's refusal of every LoRA route when `--enable-lora` is unset, which
/// Rust-server mode refuses at launch: a `LoRAUpdateOutput` with this text.
pub(crate) const LORA_DISABLED: &str =
    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA.";

/// Python answers both a missing bootstrap reply and a rejected one with
/// this message and a 400.
fn transfer_info_unavailable(rank: i64) -> CoreError {
    CoreError::InvalidArgument(format!(
        "Failed to get transfer engine info for rank {rank}"
    ))
}

impl CoreHandle {
    /// The transfer-engine info `rank` registered with this node's engine-info
    /// bootstrap server, as the bootstrap server's JSON text. A proxy, not a
    /// scheduler control: Python's handler does the same outbound GET with a
    /// five-second timeout.
    pub(crate) async fn remote_instance_transfer_engine_info(
        &self,
        rank: i64,
    ) -> Result<String, CoreError> {
        if rank < 0 {
            return Err(CoreError::InvalidArgument(
                "Missing or invalid rank parameter".into(),
            ));
        }
        let Some(base_url) = &self
            .inner
            .metadata
            .deployment_info
            .engine_info_bootstrap_url
        else {
            return Err(transfer_info_unavailable(rank));
        };
        let url = format!(
            "{}/get_transfer_engine_info",
            base_url.trim_end_matches('/')
        );
        let response = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|error| CoreError::Internal(error.to_string()))?
            .get(url)
            .query(&[("rank", rank)])
            .send()
            .await;
        let response = match response {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => {
                tracing::warn!(rank, status = %response.status(), "transfer engine info lookup refused");
                return Err(transfer_info_unavailable(rank));
            }
            Err(error) => {
                tracing::warn!(rank, %error, "transfer engine info lookup failed");
                return Err(transfer_info_unavailable(rank));
            }
        };
        let body = response
            .text()
            .await
            .map_err(|_| transfer_info_unavailable(rank))?;
        // Python returns `resp.json()`, so a body that is not JSON is the
        // same failure as no body.
        serde_json::from_str::<serde::de::IgnoredAny>(&body)
            .map_err(|_| transfer_info_unavailable(rank))?;
        Ok(body)
    }
}

/// The scheduler's `GetWeightsByNameReqOutput`: the parameter as the
/// scheduler shaped it (`List[float | List[float]]`), left untyped.
#[derive(Debug, serde::Deserialize)]
struct WeightsByNameOutput {
    #[serde(default)]
    parameter: Option<serde_json::Value>,
}

/// The `{success, message}` output every group operation returns.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct WeightGroupOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::CoreErrorKind;
    use crate::api_server::handlers::test_utils::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::config::DeploymentInfo;
    use crate::message::io_struct::WireBytes;
    use crate::message::request::RequestKind;
    use crate::message::response::ResponseItem;
    use bytes::Bytes;

    /// A parameter read keeps the scheduler's numbers as it shaped them, one
    /// entry per rank, with a missing parameter as `null`.
    #[tokio::test]
    async fn get_weights_by_name_relays_each_ranks_parameter_untyped() {
        use crate::api_server::core::fanout::testing::DummyTarget;
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
                multimodal: false,
                metrics: None,
                request_log: None,
                generate_validator: None,
                mm_limits: Vec::new(),
                metadata: CoreMetadata::default(),
                peer_endpoints: vec![DummyTarget::replying(&[1], |_| {
                    Ok(Bytes::from(
                        rmp_serde::to_vec_named(&serde_json::json!({"parameter": null})).unwrap(),
                    ))
                })],
            },
        );
        let (values, ()) = tokio::join!(handle.get_weights_by_name("w".into(), 2), async {
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            let RequestKind::Control(control) = &request.kind else {
                panic!("expected a control request");
            };
            let rmpv::Value::Array(items) =
                rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
            else {
                panic!("positional wire");
            };
            assert_eq!(items[3].as_str(), Some("w"));
            assert_eq!(items[4].as_i64(), Some(2));
            request
                .sink
                .try_send(ResponseItem::Control(Bytes::from(
                    rmp_serde::to_vec_named(&serde_json::json!({
                        "parameter": [[0.5, 1], [2.25, -3]]
                    }))
                    .unwrap(),
                )))
                .unwrap();
        },);
        assert_eq!(
            values.unwrap(),
            vec![
                serde_json::json!([[0.5, 1], [2.25, -3]]),
                serde_json::Value::Null
            ]
        );
        assert_eq!(
            handle
                .get_weights_by_name(String::new(), 100)
                .await
                .unwrap_err()
                .kind(),
            CoreErrorKind::InvalidArgument
        );
    }

    /// The proxy relays the bootstrap server's JSON as is, and maps a missing
    /// rank, a refused lookup, and an unconfigured bootstrap server onto
    /// Python's one failure message.
    #[tokio::test]
    async fn transfer_engine_info_is_proxied_from_the_bootstrap_server() {
        use axum::{Router, routing::get};
        use std::future::IntoFuture;
        let app = Router::new().route(
            "/get_transfer_engine_info",
            get(
                |axum::extract::Query(q): axum::extract::Query<
                    std::collections::HashMap<String, i64>,
                >| async move {
                    match q.get("rank") {
                        Some(0) => (
                            axum::http::StatusCode::OK,
                            r#"{"rank": 0, "remote_instance_transfer_engine_info": ["h", 1]}"#,
                        ),
                        _ => (axum::http::StatusCode::NOT_FOUND, r#"{"detail": "no"}"#),
                    }
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(axum::serve(listener, app).into_future());

        let (intake_tx, _intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = CoreHandle::new(
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
                        engine_info_bootstrap_url: Some(url),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: Vec::new(),
            },
        );
        let info = handle
            .remote_instance_transfer_engine_info(0)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&info).unwrap(),
            serde_json::json!({"rank": 0, "remote_instance_transfer_engine_info": ["h", 1]})
        );
        for rank in [-1, 7] {
            let error = handle
                .remote_instance_transfer_engine_info(rank)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), CoreErrorKind::InvalidArgument, "{rank}");
        }
        assert!(
            Harness::unbounded(8)
                .handle
                .remote_instance_transfer_engine_info(0)
                .await
                .unwrap_err()
                .to_string()
                .contains("rank 0"),
            "no bootstrap server configured"
        );
    }

    /// The group operations fan out on one rank and merge the outcome, and
    /// refuse a multi-rank deployment before reaching any scheduler.
    #[tokio::test]
    async fn group_operations_merge_outcome_and_require_one_dp_rank() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let (outcome, ()) = tokio::join!(
            handle.destroy_weights_update_group("g".into()),
            answer_local_control(
                &harness.intake_rx,
                Bytes::from(
                    rmp_serde::to_vec_named(
                        &serde_json::json!({"success": false, "message": "no such group"}),
                    )
                    .unwrap(),
                ),
            ),
        );
        assert_eq!(
            outcome.unwrap(),
            WeightGroupOutcome {
                success: false,
                message: "no such group".into()
            }
        );

        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let two_ranks = CoreHandle::new(
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
                        dp_size: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: Vec::new(),
            },
        );
        let error = two_ranks
            .send_weights_to_remote_instance("h".into(), "1".into(), "g".into())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::InvalidArgument);
        assert!(intake_rx.try_recv().is_err(), "refused before any rank");
    }

    /// The wire for a control request the local scheduler received: the
    /// positional header items after the `BaseReq` preamble.
    fn wire_items(request: &crate::message::request::Request) -> Vec<rmpv::Value> {
        let RequestKind::Control(control) = &request.kind else {
            panic!("expected a control request");
        };
        let rmpv::Value::Array(items) =
            rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
        else {
            panic!("positional wire");
        };
        items
    }

    fn status_reply(success: bool, message: &str) -> Bytes {
        Bytes::from(
            rmp_serde::to_vec_named(&serde_json::json!({"success": success, "message": message}))
                .unwrap(),
        )
    }

    /// A disk update fills the launch's `load_format`, carries every field
    /// in Python's order, merges one `num_paused_requests` per rank, and
    /// appends the weight-version note on success only.
    #[tokio::test]
    async fn update_weights_from_disk_fills_the_launch_load_format_and_merges_ranks() {
        use crate::api_server::core::fanout::testing::DummyTarget;
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let peer_reply = |success: bool| {
            Bytes::from(
                rmp_serde::to_vec_named(&serde_json::json!({
                    "success": success, "message": "peer", "num_paused_requests": 2
                }))
                .unwrap(),
            )
        };
        let handle = CoreHandle::new(
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
                    load_format: Some("safetensors".into()),
                    deployment_info: DeploymentInfo {
                        dp_size: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: vec![DummyTarget::replying(&[1], move |_| Ok(peer_reply(true)))],
            },
        );
        let update = DiskWeightUpdate {
            model_path: "/models/new".into(),
            weight_version: Some("v2".into()),
            flush_cache: true,
            ..Default::default()
        };
        let (outcome, items) = tokio::join!(handle.update_weights_from_disk(update), async {
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            let items = wire_items(&request);
            request
                .sink
                .try_send(ResponseItem::Control(Bytes::from(
                    rmp_serde::to_vec_named(&serde_json::json!({
                        "success": true, "message": "Succeeded to update model weights.",
                        "num_paused_requests": 0
                    }))
                    .unwrap(),
                )))
                .unwrap();
            items
        });
        // [tag, rid, ipc, model_path, load_format, abort_all, weight_version,
        //  is_async, torch_empty_cache, keep_pause, recapture, token_step,
        //  flush_cache, manifest]
        assert_eq!(items[0].as_str(), Some("UpdateWeightFromDiskReqInput"));
        assert_eq!(items[3].as_str(), Some("/models/new"));
        assert_eq!(items[4].as_str(), Some("safetensors"), "launch load_format");
        assert_eq!(items[6].as_str(), Some("v2"));
        assert_eq!(items[12].as_bool(), Some(true));
        assert!(items[13].is_nil());
        assert_eq!(
            outcome.unwrap(),
            DiskUpdateOutcome {
                success: true,
                message: "Succeeded to update model weights. | peer Weight version updated to v2."
                    .into(),
                num_paused_requests: vec![0, 2],
            }
        );
        assert!(
            !handle.inner.frontend.is_paused(),
            "released after the update"
        );
        assert!(handle.inner.frontend.is_stale(), "the facts changed");

        let error = handle
            .update_weights_from_disk(DiskWeightUpdate::default())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::InvalidArgument);
    }

    /// While an update runs, generates wait at admission and resume once
    /// the scheduler answered; a deployment paused before stays paused.
    #[tokio::test]
    async fn an_update_holds_admission_until_the_scheduler_answers() {
        use crate::message::request::GenerateRequest;
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let update =
            tokio::spawn(async move { handle.begin_weight_update("all".into()).await.unwrap() });
        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert_eq!(wire_items(&request)[3].as_str(), Some("all"));
        assert!(harness.handle.inner.frontend.is_paused());
        let gen_handle = harness.handle.clone();
        let parked = tokio::spawn(async move {
            gen_handle
                .generate(
                    GenerateRequest {
                        input_ids: Some(vec![1]),
                        ..Default::default()
                    },
                    crate::api_server::core::ReceivedAs::without_headers(),
                )
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert!(!parked.is_finished(), "generate waits for the update");
        assert!(harness.intake_rx.try_recv().is_err());
        request
            .sink
            .try_send(ResponseItem::Control(status_reply(true, "Success")))
            .unwrap();
        assert_eq!(
            update.await.unwrap(),
            WeightUpdateOutcome {
                success: true,
                message: "Success".into()
            }
        );
        let released = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(matches!(released.kind, RequestKind::Generate(_)));
        assert!(!harness.handle.inner.frontend.is_paused());
        drop(parked);

        // Paused by the client already: the update leaves the pause in place.
        harness.handle.inner.frontend.set_paused(true);
        let handle = harness.handle.clone();
        let (outcome, ()) = tokio::join!(
            handle.end_weight_update(),
            answer_local_control(&harness.intake_rx, status_reply(false, "no session")),
        );
        assert!(!outcome.unwrap().success);
        assert!(
            harness.handle.inner.frontend.is_paused(),
            "kept the client's pause"
        );
    }

    /// The session updates refuse a multi-rank deployment before any
    /// scheduler sees them (IPC answers Python's wrapped failure instead), an
    /// unknown selector is a 400, and the plugin merge keeps the first
    /// rank's counts on success only.
    #[tokio::test]
    async fn session_updates_require_one_rank_and_plugin_merges_counts() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let (outcome, items) = tokio::join!(
            handle.plugin_weight_update(PluginWeightUpdate {
                handler: "p2p".into(),
                payload: serde_json::json!({"bucket": 3})
                    .as_object()
                    .cloned()
                    .unwrap(),
                selector: "all".into(),
                ..Default::default()
            }),
            async {
                let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
                let items = wire_items(&request);
                request
                    .sink
                    .try_send(ResponseItem::Control(Bytes::from(
                        rmp_serde::to_vec_named(&serde_json::json!({
                            "success": true, "message": "ok", "received_tensor_counts": [4, 4]
                        }))
                        .unwrap(),
                    )))
                    .unwrap();
                items
            },
        );
        assert_eq!(items[3].as_str(), Some("p2p"));
        assert_eq!(
            items[4],
            rmpv::Value::Map(vec![("bucket".into(), 3.into())]),
            "the payload reaches the scheduler as a map"
        );
        assert_eq!(
            outcome.unwrap(),
            PluginUpdateOutcome {
                success: true,
                message: "ok".into(),
                received_tensor_counts: vec![4, 4],
            }
        );
        let handle = harness.handle.clone();
        let (outcome, ()) = tokio::join!(
            handle.update_weights_from_tensor(TensorWeightUpdate {
                serialized_named_tensors: vec![WireBytes(vec![0x80, 0x05])],
                selector: "draft".into(),
                weight_version: Some("v3".into()),
                ..Default::default()
            }),
            async {
                let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
                let items = wire_items(&request);
                assert_eq!(
                    items[3],
                    rmpv::Value::Array(vec![rmpv::Value::Binary(vec![0x80, 0x05])]),
                    "payloads ride the wire as bin"
                );
                request
                    .sink
                    .try_send(ResponseItem::Control(status_reply(false, "no session")))
                    .unwrap();
            },
        );
        assert_eq!(
            outcome.unwrap(),
            WeightUpdateOutcome {
                success: false,
                message: "no session".into()
            },
            "no version note on a failure"
        );
        assert_eq!(
            harness
                .handle
                .begin_weight_update("sideways".into())
                .await
                .unwrap_err()
                .kind(),
            CoreErrorKind::InvalidArgument
        );

        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let two_ranks = CoreHandle::new(
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
                        dp_size: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: Vec::new(),
            },
        );
        let error = two_ranks
            .update_weights_from_distributed(DistributedWeightUpdate {
                selector: "all".into(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::InvalidArgument);
        let outcome = two_ranks
            .update_weights_from_ipc(IpcWeightUpdate {
                zmq_handles: [("0".to_owned(), "ipc:///x".to_owned())].into(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!outcome.success && outcome.message.starts_with("IPC weight update failed"));
        assert!(intake_rx.try_recv().is_err(), "refused before any rank");
    }
}
