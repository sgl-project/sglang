//! Scheduler control operations fanned out to every DP rank, and the relay
//! that serves another listener's request on the local scheduler.

use crate::message::ids::Rid;
use crate::message::io_struct::{
    CheckWeightsReqInput, ControlRequest, ExpertDistributionAction, ExpertDistributionReq,
    FlushCacheReqInput, ProfileOptions, ProfileReq, ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput, SetInternalStateReq, SlowDownReqInput,
};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::message::types::ForwardedControlReq;
use crate::utils::environ;

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

    /// Start a profiler capture on every DP rank. Fills the defaults Python's
    /// `TokenizerManager.start_profile` fills: `with_stack` and `record_shapes`
    /// are on unless the client or the env flag says no, and a missing
    /// `profile_id` is the current time. Succeeds only if every rank did.
    pub(crate) async fn start_profile(
        &self,
        mut options: ProfileOptions,
    ) -> Result<ProfileOutcome, CoreError> {
        options.with_stack = Some(
            options.with_stack != Some(false)
                && environ::env_bool("SGLANG_PROFILE_WITH_STACK", true),
        );
        options.record_shapes = Some(
            options.record_shapes != Some(false)
                && environ::env_bool("SGLANG_PROFILE_RECORD_SHAPES", true),
        );
        if options.profile_id.is_none() {
            // Python's `str(time.time())`: seconds since the epoch, fractional.
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or_default();
            options.profile_id = Some(format!("{now}"));
        }
        self.profile(&|| {
            ControlRequest::ProfileReq(ProfileReq::start(Rid::new().to_string(), options.clone()))
        })
        .await
    }

    /// Stop the profiler capture on every DP rank; succeeds only if every rank did.
    pub(crate) async fn stop_profile(&self) -> Result<ProfileOutcome, CoreError> {
        self.profile(&|| ControlRequest::ProfileReq(ProfileReq::stop(Rid::new().to_string())))
            .await
    }

    async fn profile(&self, build: &RequestBuilder<'_>) -> Result<ProfileOutcome, CoreError> {
        let outcomes = self
            .control(build)
            .await
            .decode::<ProfileOutcome>("profile")?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        Ok(ProfileOutcome { success, message })
    }

    /// Start, stop, or dump the expert-distribution recorder on every DP rank.
    pub(crate) async fn expert_distribution_record(
        &self,
        action: ExpertDistributionAction,
    ) -> Result<(), CoreError> {
        self.ack("expert-distribution", &|| {
            ControlRequest::ExpertDistributionReq(ExpertDistributionReq::new(
                Rid::new().to_string(),
                action,
            ))
        })
        .await
    }

    /// Sleep `forward_sleep_time` seconds before every forward on every DP
    /// rank; `None` or non-positive clears it. Testing only.
    pub(crate) async fn slow_down(&self, forward_sleep_time: Option<f64>) -> Result<(), CoreError> {
        self.ack("slow-down", &|| {
            ControlRequest::SlowDownReqInput(SlowDownReqInput::new(
                Rid::new().to_string(),
                forward_sleep_time,
            ))
        })
        .await
    }

    /// Free the GPU memory of the tagged regions on every DP rank.
    pub(crate) async fn release_memory_occupation(
        &self,
        tags: Option<Vec<String>>,
    ) -> Result<(), CoreError> {
        self.ack("release-memory-occupation", &|| {
            ControlRequest::ReleaseMemoryOccupationReqInput(ReleaseMemoryOccupationReqInput::new(
                Rid::new().to_string(),
                tags.clone(),
            ))
        })
        .await
    }

    /// Reclaim the GPU memory of the tagged regions on every DP rank.
    pub(crate) async fn resume_memory_occupation(
        &self,
        tags: Option<Vec<String>>,
    ) -> Result<(), CoreError> {
        self.ack("resume-memory-occupation", &|| {
            ControlRequest::ResumeMemoryOccupationReqInput(ResumeMemoryOccupationReqInput::new(
                Rid::new().to_string(),
                tags.clone(),
            ))
        })
        .await
    }

    /// A control operation whose scheduler output carries no fields: done
    /// once every rank acknowledged.
    async fn ack(&self, what: &str, build: &RequestBuilder<'_>) -> Result<(), CoreError> {
        self.control(build).await.decode::<Ack>(what).map(|_| ())
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

/// A scheduler output with no fields of its own (`BaseReq` only).
#[derive(Debug, serde::Deserialize)]
struct Ack {}

/// The scheduler's `ProfileReqOutput`.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct ProfileOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
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
impl CoreHandle {
    async fn into_owned_expert(self, action: ExpertDistributionAction) -> Result<(), CoreError> {
        self.expert_distribution_record(action).await
    }
    async fn into_owned_slow_down(self, t: Option<f64>) -> Result<(), CoreError> {
        self.slow_down(t).await
    }
    async fn into_owned_release(self, tags: Option<Vec<String>>) -> Result<(), CoreError> {
        self.release_memory_occupation(tags).await
    }
    async fn into_owned_resume(self, tags: Option<Vec<String>>) -> Result<(), CoreError> {
        self.resume_memory_occupation(tags).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::test_utils::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::io_struct::GetInternalStateReq;
    use crate::message::request::Request;
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

    /// Start and stop differ only in the wire's `req_type`; start fills the
    /// env-driven defaults and mints a profile id, stop sends a bare request.
    #[tokio::test]
    async fn profile_requests_carry_python_defaults_and_the_req_type() {
        use crate::message::io_struct::{PROFILE_REQ_START, PROFILE_REQ_STOP};
        let harness = Harness::unbounded(8);
        let ok = Bytes::from(
            rmp_serde::to_vec_named(&serde_json::json!({"success": true, "message": ""})).unwrap(),
        );
        let wire = |request: &Request| -> Vec<rmpv::Value> {
            let RequestKind::Control(control) = &request.kind else {
                panic!("expected a control request");
            };
            let rmpv::Value::Array(items) =
                rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
            else {
                panic!("positional wire");
            };
            items
        };

        let handle = harness.handle.clone();
        let (outcome, items) = tokio::join!(
            handle.start_profile(ProfileOptions {
                num_steps: Some(5),
                with_stack: Some(false),
                ..Default::default()
            }),
            async {
                let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
                let items = wire(&request);
                request
                    .sink
                    .try_send(ResponseItem::Control(ok.clone()))
                    .unwrap();
                items
            },
        );
        assert!(outcome.unwrap().success);
        // [tag, rid, http_worker_ipc, req_type, output_dir, start_step, num_steps,
        //  activities, profile_by_stage, with_stack, record_shapes, profile_id, ...]
        assert_eq!(items[0].as_str(), Some("ProfileReq"));
        assert_eq!(items[3].as_u64(), Some(u64::from(PROFILE_REQ_START)));
        assert_eq!(items[6].as_i64(), Some(5));
        assert_eq!(items[9].as_bool(), Some(false), "the client's false wins");
        assert_eq!(items[10].as_bool(), Some(true), "record_shapes defaults on");
        assert!(
            items[11]
                .as_str()
                .is_some_and(|id| id.parse::<f64>().is_ok()),
            "a minted profile_id is a timestamp: {:?}",
            items[11]
        );

        let handle = harness.handle.clone();
        let (outcome, items) = tokio::join!(handle.stop_profile(), async {
            let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
            let items = wire(&request);
            request
                .sink
                .try_send(ResponseItem::Control(ok.clone()))
                .unwrap();
            items
        });
        assert!(outcome.unwrap().success);
        assert_eq!(items[3].as_u64(), Some(u64::from(PROFILE_REQ_STOP)));
        assert!(items[11].is_nil(), "stop mints no profile_id");
    }

    /// The acknowledged operations encode their one payload field and treat
    /// the scheduler's bare `BaseReq` reply as done.
    #[tokio::test]
    async fn acknowledged_operations_encode_their_payload_and_accept_a_bare_reply() {
        let harness = Harness::unbounded(8);
        let bare = Bytes::from(
            rmp_serde::to_vec_named(&serde_json::json!({"rid": "x", "http_worker_ipc": null}))
                .unwrap(),
        );
        let field = |request: &Request| -> rmpv::Value {
            let RequestKind::Control(control) = &request.kind else {
                panic!("expected a control request");
            };
            let rmpv::Value::Array(mut items) =
                rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
            else {
                panic!("positional wire");
            };
            items.remove(3) // the first field after the BaseReq preamble
        };
        type Call =
            std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), CoreError>> + Send>>;
        let handle = harness.handle.clone();
        let cases: Vec<(&str, Call, rmpv::Value)> = vec![
            (
                "ExpertDistributionReq",
                Box::pin(
                    handle
                        .clone()
                        .into_owned_expert(ExpertDistributionAction::Dump),
                ),
                3u8.into(),
            ),
            (
                "SlowDownReqInput",
                Box::pin(handle.clone().into_owned_slow_down(Some(0.5))),
                0.5f64.into(),
            ),
            (
                "ReleaseMemoryOccupationReqInput",
                Box::pin(
                    handle
                        .clone()
                        .into_owned_release(Some(vec!["weights".into()])),
                ),
                rmpv::Value::Array(vec!["weights".into()]),
            ),
            (
                "ResumeMemoryOccupationReqInput",
                Box::pin(handle.clone().into_owned_resume(None)),
                rmpv::Value::Nil,
            ),
        ];
        for (tag, call, expected) in cases {
            let (outcome, value) = tokio::join!(call, async {
                let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
                let RequestKind::Control(control) = &request.kind else {
                    panic!("expected a control request");
                };
                assert_eq!(
                    control.encode().unwrap().get(1).map(|_| ()),
                    Some(()),
                    "{tag}: encoded"
                );
                let value = field(&request);
                request
                    .sink
                    .try_send(ResponseItem::Control(bare.clone()))
                    .unwrap();
                value
            });
            outcome.unwrap_or_else(|error| panic!("{tag}: {error}"));
            assert_eq!(value, expected, "{tag}");
        }
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
