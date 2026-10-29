//! Scheduler control operations (flush cache, weights check, profiling)
//! fanned out to every DP rank, and the relay that serves another listener's
//! request on the local scheduler.

use crate::message::ids::Rid;
use crate::message::io_struct::{
    CheckWeightsReqInput, ControlRequest, FlushCacheReqInput, ProfileOptions, ProfileReq,
};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::message::types::ForwardedControlReq;

use super::handle::{CoreHandle, translate_runtime_error};
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout::{self, RequestBuilder};
use crate::utils::environ;

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

    /// Start profiling on every DP rank's scheduler (each broadcasts to its
    /// TP ranks), resolving `options` as Python's
    /// `TokenizerManager.start_profile` does (see [`resolve_start_profile`]).
    pub(crate) async fn start_profile(
        &self,
        options: ProfileOptions,
    ) -> Result<ProfileOutcome, CoreError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let options = resolve_start_profile(options, environ::get_bool_env_var, now);
        self.profile(&|| ProfileReq::start(Rid::new().to_string(), options.clone()))
            .await
    }

    /// Stop profiling on every DP rank and write their traces.
    pub(crate) async fn stop_profile(&self) -> Result<ProfileOutcome, CoreError> {
        self.profile(&|| ProfileReq::stop(Rid::new().to_string()))
            .await
    }

    /// Fan one `ProfileReq` out to every rank; succeeds only if all ranks did
    /// (a refusal — already profiling, nothing to stop — carries its reason).
    async fn profile(
        &self,
        build: &(dyn Fn() -> ProfileReq + Sync),
    ) -> Result<ProfileOutcome, CoreError> {
        let outcomes = self
            .control(&|| ControlRequest::ProfileReq(build()))
            .await
            .decode::<ProfileOutcome>("profile")?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        Ok(ProfileOutcome { success, message })
    }

    /// Deliver one control request to every DP rank of the deployment.
    pub(super) async fn control(&self, build: &RequestBuilder<'_>) -> fanout::Replies {
        fanout::fan_out(
            self.inner.dp_rank,
            self.submit_control_req(build()),
            &self.inner.peer_endpoints,
            build,
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

/// The scheduler's `FlushCacheReqOutput`: whether the flush ran, and why not.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct FlushCacheOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

/// The scheduler's `ProfileReqOutput`: whether the profiler call succeeded,
/// and why not.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct ProfileOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

/// Python's `TokenizerManager.start_profile` defaults: stacks and shapes are
/// recorded unless the body or `SGLANG_PROFILE_WITH_STACK` /
/// `SGLANG_PROFILE_RECORD_SHAPES` (read by `env_flag`, default true) turns
/// them off, and a missing or empty `profile_id` becomes `str(time.time())`
/// (`now`, in seconds) — the trace file name prefix.
fn resolve_start_profile(
    mut options: ProfileOptions,
    env_flag: impl Fn(&str, bool) -> bool,
    now: f64,
) -> ProfileOptions {
    options.with_stack =
        Some(options.with_stack != Some(false) && env_flag("SGLANG_PROFILE_WITH_STACK", true));
    options.record_shapes = Some(
        options.record_shapes != Some(false) && env_flag("SGLANG_PROFILE_RECORD_SHAPES", true),
    );
    if options.profile_id.as_deref().is_none_or(str::is_empty) {
        // `{:?}` prints the shortest round-trip form with a fractional part,
        // as Python's `str(float)` does.
        options.profile_id = Some(format!("{now:?}"));
    }
    options
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

    /// `start_profile` resolves the body as Python's
    /// `TokenizerManager.start_profile`: a body `false` or an env knob not
    /// `true`/`1` turns stacks/shapes off, and a missing or empty
    /// `profile_id` becomes `str(time.time())`.
    #[test]
    fn start_profile_resolves_python_defaults() {
        for (body, env, want) in [
            (None, None, true),
            (Some(true), None, true),
            (Some(false), None, false),
            (None, Some(false), false),
            (Some(true), Some(false), false),
        ] {
            let resolved = resolve_start_profile(
                ProfileOptions {
                    with_stack: body,
                    record_shapes: body,
                    ..Default::default()
                },
                |name, default| {
                    assert!(default, "{name} defaults to true");
                    env.unwrap_or(default)
                },
                1.0,
            );
            assert_eq!(resolved.with_stack, Some(want), "{body:?} {env:?}");
            assert_eq!(resolved.record_shapes, Some(want), "{body:?} {env:?}");
        }

        for (profile_id, now, want) in [
            (None, 1729771234.0, "1729771234.0"),
            (Some(""), 1729771234.5678904, "1729771234.5678904"),
            (Some("run-1"), 1.0, "run-1"),
        ] {
            let resolved = resolve_start_profile(
                ProfileOptions {
                    profile_id: profile_id.map(str::to_owned),
                    ..Default::default()
                },
                |_, default| default,
                now,
            );
            assert_eq!(resolved.profile_id.as_deref(), Some(want));
        }
    }

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
                multimodal: false,
                metrics: None,
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
