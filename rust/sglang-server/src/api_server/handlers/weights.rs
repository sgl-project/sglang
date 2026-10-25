//! Weight management, the state-free subset: process-group setup,
//! remote-instance sends, and parameter reads. The weight updates themselves
//! wait on the frontend holding the update lock on every listener.

use crate::message::ids::Rid;
use crate::message::io_struct::{
    ControlRequest, DestroyWeightsUpdateGroupReqInput, GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput, InitWeightsUpdateGroupReqInput,
    SendWeightsToRemoteInstanceReqInput, WeightsSendGroup, WeightsUpdateGroup,
};

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
}
