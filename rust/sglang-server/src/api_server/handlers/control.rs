//! Scheduler control operations fanned out to every DP rank, and the relay
//! that serves another listener's request on the local scheduler.

use crate::message::ids::Rid;
use crate::message::io_struct::{
    CheckWeightsReqInput, ControlRequest, FlushCacheReqInput, SetInternalStateReq,
};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::message::types::ForwardedControlReq;

use super::handle::{CoreHandle, translate_runtime_error};
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout::{self, RequestBuilder};

impl CoreHandle {
    /// Flush every DP rank's caches; succeeds only if all ranks flushed (Python checks rank 0 only).
    pub(crate) async fn flush_cache(&self, timeout_s: f64) -> Result<FlushCacheOutcome, CoreError> {
        let outcomes = self
            .control(&|| {
                ControlRequest::FlushCacheReqInput(FlushCacheReqInput::new(
                    Rid::new().to_string(),
                    Some(timeout_s),
                ))
            })
            .await
            .decode::<FlushCacheOutcome>("flush-cache")?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        Ok(FlushCacheOutcome { success, message })
    }

    /// Override numeric scheduler knobs on every DP rank; one `updated` flag per
    /// rank in rank order, as Python's `set_internal_state` returns them.
    /// `server_args` is the client's JSON object, validated here only as far as
    /// Python's request type does (an object of numbers); the scheduler decides
    /// which keys it accepts.
    pub(crate) async fn set_internal_state(
        &self,
        server_args: Option<&str>,
    ) -> Result<Vec<bool>, CoreError> {
        let server_args = parse_server_args(server_args)?;
        let outputs = self
            .control(&|| {
                ControlRequest::SetInternalStateReq(SetInternalStateReq::new(
                    Rid::new().to_string(),
                    server_args.clone(),
                ))
            })
            .await
            .decode::<SetInternalStateOutput>("set-internal-state")?;
        Ok(outputs.into_iter().map(|output| output.updated).collect())
    }

    /// Run the weight checker on every DP rank; merge as Python's `check_weights` does.
    pub(crate) async fn check_weights(
        &self,
        action: String,
        allow_quant_error: bool,
        skip_tensor_list: Option<Vec<String>>,
        selector: String,
    ) -> Result<CheckWeightsOutcome, CoreError> {
        let outcomes = self
            .control(&|| {
                ControlRequest::CheckWeightsReqInput(CheckWeightsReqInput::new(
                    Rid::new().to_string(),
                    action.clone(),
                    allow_quant_error,
                    skip_tensor_list.clone(),
                    selector.clone(),
                ))
            })
            .await
            .decode::<CheckWeightsOutcome>("weights-check")?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        let mut payload: Option<Vec<_>> = None;
        for records in outcomes.into_iter().filter_map(|outcome| outcome.payload) {
            payload.get_or_insert_with(Vec::new).extend(records);
        }
        Ok(CheckWeightsOutcome {
            success,
            message,
            payload,
        })
    }

    /// Deliver one control request to every DP rank of the deployment.
    pub(super) async fn control(&self, build: &RequestBuilder<'_>) -> fanout::Replies {
        fanout::fan_out(
            self.inner.dp_rank,
            self.submit_control_req(build()),
            &self.inner.peer_endpoints,
            build,
            None,
        )
        .await
    }

    /// Serve a peer's control request on the local scheduler only; never fan out again.
    pub(crate) async fn forward_control_req(
        &self,
        header: bytes::Bytes,
    ) -> Result<bytes::Bytes, CoreError> {
        let forwarded = ForwardedControlReq::parse(header)
            .map_err(|error| CoreError::InvalidArgument(error.to_string()))?;
        self.submit_control_req(ControlRequest::Forwarded(forwarded))
            .await
    }

    /// Deliver one control request to the local scheduler only.
    async fn submit_control_req(&self, request: ControlRequest) -> Result<bytes::Bytes, CoreError> {
        let mut call = self.submit(RequestKind::Control(Box::new(request))).await?;
        match call.recv().await {
            Some(ResponseItem::Control(bytes)) => Ok(bytes),
            Some(ResponseItem::Error(error)) => Err(translate_runtime_error(error)),
            Some(_) => Err(CoreError::InvalidResponse(
                "unexpected generation output for control request".into(),
            )),
            None => Err(CoreError::ResponseTruncated),
        }
    }
}

/// Python's `SetInternalStateReq.server_args` contract: a JSON object whose
/// values are numbers. Anything else is the client's error, not the scheduler's.
fn parse_server_args(
    text: Option<&str>,
) -> Result<serde_json::Map<String, serde_json::Value>, CoreError> {
    let Some(text) = text else {
        return Err(CoreError::InvalidArgument(
            "missing field `server_args`".into(),
        ));
    };
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|error| CoreError::InvalidArgument(format!("invalid `server_args`: {error}")))?;
    let serde_json::Value::Object(map) = value else {
        return Err(CoreError::InvalidArgument(
            "`server_args` must be a JSON object".into(),
        ));
    };
    if let Some((key, _)) = map.iter().find(|(_, value)| !value.is_number()) {
        return Err(CoreError::InvalidArgument(format!(
            "`server_args.{key}` must be a number"
        )));
    }
    Ok(map)
}

/// The scheduler's `SetInternalStateReqOutput`.
#[derive(Debug, serde::Deserialize, PartialEq)]
struct SetInternalStateOutput {
    updated: bool,
}

/// The scheduler's `FlushCacheReqOutput`: whether the flush ran, and why not.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct FlushCacheOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

/// The scheduler's `CheckWeightsReqOutput`: whether every rank succeeded, why
/// not, and one checksum record per rank (`None` when no rank produced one).
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct CheckWeightsOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
    #[serde(default)]
    pub(crate) payload: Option<Vec<serde_json::Map<String, serde_json::Value>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::test_utils::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::io_struct::GetInternalStateReq;
    use bytes::Bytes;

    /// A multi-rank operation reaches the local scheduler once and every
    /// target once, and its outcome is the merge of all of them: one busy
    /// rank fails the flush, with its reason.
    #[tokio::test]
    async fn flush_cache_merges_the_local_rank_with_every_target() {
        use crate::api_server::core::fanout::testing::{DummyTarget, status_bytes};
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
                peer_endpoints: vec![DummyTarget::replying(&[1, 2], |rank| {
                    Ok(if rank == 2 {
                        status_bytes(false, "peer busy")
                    } else {
                        status_bytes(true, "")
                    })
                })],
            },
        );

        let (outcome, ()) = tokio::join!(
            handle.flush_cache(1.0),
            answer_local_control(&intake_rx, status_bytes(true, "")),
        );
        let outcome = outcome.unwrap();
        assert!(!outcome.success);
        assert_eq!(outcome.message, "peer busy");
        assert!(
            intake_rx.try_recv().is_err(),
            "one fan-out submits one local request"
        );
    }

    /// Each rank answers for itself: the flags come back in rank order, and a
    /// body that is not an object of numbers is rejected before any rank sees it.
    #[tokio::test]
    async fn set_internal_state_collects_one_flag_per_rank() {
        use crate::api_server::core::CoreErrorKind;
        use crate::api_server::core::fanout::testing::DummyTarget;
        let updated = |flag: bool| {
            Bytes::from(rmp_serde::to_vec_named(&serde_json::json!({ "updated": flag })).unwrap())
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
                metadata: CoreMetadata::default(),
                peer_endpoints: vec![DummyTarget::replying(&[1], move |_| Ok(updated(false)))],
            },
        );

        let (flags, ()) = tokio::join!(
            handle.set_internal_state(Some(r#"{"pp_max_micro_batch_size": 4}"#)),
            async {
                let request = accept_intake(intake_rx.recv_async().await.unwrap());
                let RequestKind::Control(control) = &request.kind else {
                    panic!("expected a control request");
                };
                let header = rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap();
                let rmpv::Value::Array(items) = header else {
                    panic!("positional wire");
                };
                assert_eq!(items[0].as_str(), Some("SetInternalStateReq"));
                assert_eq!(
                    items[3],
                    rmpv::Value::Map(vec![("pp_max_micro_batch_size".into(), 4.into())]),
                    "the object reaches the scheduler as a msgpack map with its integer intact"
                );
                request
                    .sink
                    .try_send(ResponseItem::Control(updated(true)))
                    .unwrap();
            },
        );
        assert_eq!(flags.unwrap(), [true, false]);

        for bad in [None, Some("[]"), Some("{\"k\": \"v\"}"), Some("not json")] {
            let error = handle.set_internal_state(bad).await.unwrap_err();
            assert_eq!(error.kind(), CoreErrorKind::InvalidArgument, "{bad:?}");
        }
        assert!(
            intake_rx.try_recv().is_err(),
            "rejected bodies reach no rank"
        );
    }

    #[tokio::test]
    async fn control_operation_returns_transport_neutral_payload() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let task = tokio::spawn(async move {
            handle
                .submit_control_req(ControlRequest::GetInternalStateReq(
                    GetInternalStateReq::new("control-id".into()),
                ))
                .await
        });

        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert_eq!(request.rid.as_str(), "control-id");
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::from_static(b"payload")))
            .unwrap();
        assert_eq!(task.await.unwrap().unwrap(), Bytes::from_static(b"payload"));
        assert!(harness.abort_rx.try_recv().is_err());
    }
}
