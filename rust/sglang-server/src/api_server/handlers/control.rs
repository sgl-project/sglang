//! Scheduler control operations (flush cache, weights check, profiling)
//! fanned out to every DP rank, and the relay that serves another listener's
//! request on the local scheduler.

use std::time::Duration;

use crate::message::ids::Rid;
use crate::message::io_struct::{
    AbortReq, CheckWeightsReqInput, ControlRequest, FlushCacheReqInput, ProfileOptions, ProfileReq,
};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::message::types::ForwardedControlReq;

use super::generate::AbortEnding;
use super::handle::{CoreHandle, translate_runtime_error};
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout::{self, RankReply, RequestBuilder};
use crate::utils::environ;

/// One relayed abort, connect to reply: the client was answered already,
/// and a peer that has not answered by then is logged as missed.
const PEER_ABORT_TIMEOUT: Duration = Duration::from_secs(5);
/// Relayed aborts in flight at once on one listener, across every abort;
/// the rest wait, within their [`PEER_ABORT_TIMEOUT`].
pub(super) const PEER_ABORT_MAX_IN_FLIGHT: usize = 64;

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
        names: Option<Vec<String>>,
    ) -> Result<CheckWeightsOutcome, CoreError> {
        let outcomes = self
            .control(&|| {
                ControlRequest::CheckWeightsReqInput(CheckWeightsReqInput::new(
                    Rid::new().to_string(),
                    action.clone(),
                    allow_quant_error,
                    skip_tensor_list.clone(),
                    selector.clone(),
                    names.clone(),
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

    /// Serve a peer's control request on the local scheduler only; never fan
    /// out again. An abort is the one message served here instead of there:
    /// the public rid it names is known to this listener's cancellation
    /// registry, not to its scheduler, which sees internal ids, so it ends
    /// this listener's own generations as `/abort_request` does, and the
    /// reply is whether any matched (see [`CoreHandle::abort_peers`]).
    pub(crate) async fn forward_control_req(
        &self,
        header: bytes::Bytes,
    ) -> Result<bytes::Bytes, CoreError> {
        let forwarded = ForwardedControlReq::parse(header)
            .map_err(|error| CoreError::InvalidArgument(error.to_string()))?;
        if let Some(abort) = AbortReq::decode(&forwarded.header)
            .map_err(|error| CoreError::InvalidArgument(error.to_string()))?
        {
            let rid = (!abort.abort_all()).then(|| abort.get_rid());
            let matched = self.abort(rid, AbortEnding::AbortFinishReason);
            return rmp_serde::to_vec(&matched)
                .map(bytes::Bytes::from)
                .map_err(|error| CoreError::Internal(error.to_string()));
        }
        self.submit_control_req(ControlRequest::Forwarded(forwarded))
            .await
    }

    /// Relay an abort `/abort_request` served locally to every other DP
    /// rank's listener, as the scheduler's own `AbortReq` for the same public
    /// `rid` (every request with `abort_all`). Only the listener of DP rank
    /// 0, the one serving the deployment's base URL, relays: each listener
    /// feeds only its own DP group and tracks only its own generations, so
    /// an abort sent to the base URL reaches the rest only this way; each
    /// serves it on its own registry ([`CoreHandle::forward_control_req`])
    /// and never relays it again. A no-op abort (no rid, not `abort_all`) is
    /// not relayed, as it is not served. Best effort: the caller has already
    /// answered its client; each post is bounded by [`PEER_ABORT_TIMEOUT`],
    /// at most [`PEER_ABORT_MAX_IN_FLIGHT`] are in flight at once, and one
    /// line logs the ranks an abort did not reach.
    pub(crate) async fn abort_peers(&self, rid: String, abort_all: bool) {
        let peers = &self.inner.peer_endpoints;
        if self.inner.dp_rank != 0 || peers.is_empty() || (rid.is_empty() && !abort_all) {
            return;
        }
        // `abort_all` names no request, but the relay routes its reply by the
        // rid; mint one, as every fanned-out control request does.
        let rid = if rid.is_empty() {
            Rid::new().to_string()
        } else {
            rid
        };
        let build = || ControlRequest::AbortReq(AbortReq::new(rid.clone(), abort_all));
        let posts = peers.iter().map(|peer| async {
            let post = async {
                let _permit = self
                    .inner
                    .peer_abort_permits
                    .acquire()
                    .await
                    .expect("the abort permits are never closed");
                peer.control(&build).await
            };
            match tokio::time::timeout(PEER_ABORT_TIMEOUT, post).await {
                Ok(replies) => replies
                    .into_iter()
                    .filter_map(|RankReply { rank, outcome }| {
                        outcome
                            .err()
                            .map(|error| format!("dp rank {rank}: {error}"))
                    })
                    .collect::<Vec<_>>(),
                Err(_) => peer
                    .ranks()
                    .iter()
                    .map(|rank| format!("dp rank {rank}: timed out after {PEER_ABORT_TIMEOUT:?}"))
                    .collect(),
            }
        });
        let missed: Vec<String> = futures::future::join_all(posts)
            .await
            .into_iter()
            .flatten()
            .collect();
        if !missed.is_empty() {
            tracing::warn!(
                rid = %rid,
                abort_all,
                "abort did not reach {} of {} DP peers: {}",
                missed.len(),
                peers.len(),
                missed.join("; ")
            );
        }
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

    /// A relayed `AbortReq` ends this listener's own generations under the
    /// sender's public rid, as `/abort_request` would, without reaching the
    /// scheduler; the reply says whether anything matched.
    #[tokio::test]
    async fn a_forwarded_abort_is_served_on_the_local_registry() {
        use crate::message::request::GenerateRequest;
        use crate::tokenizer_manager::wiring::AbortSource;
        let harness = Harness::unbounded(8);
        let call = harness
            .handle
            .generate(GenerateRequest {
                rid: Rid::from_client("client-rid"),
                ..Default::default()
            })
            .await
            .unwrap();
        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        for (abort, expected) in [
            (AbortReq::new("other-rid".into(), false), false),
            (AbortReq::new("client-rid".into(), false), true),
        ] {
            let reply = harness
                .handle
                .forward_control_req(abort.encode().unwrap())
                .await
                .unwrap();
            assert_eq!(rmp_serde::from_slice::<bool>(&reply).unwrap(), expected);
        }
        assert!(matches!(
            harness.abort_rx.try_recv(),
            Ok(AbortSource::Request(rid)) if rid == request.rid
        ));
        assert!(
            harness.intake_rx.try_recv().is_err(),
            "an abort is not a control request for the scheduler"
        );
        drop(call);
        // `abort_all` rides a minted rid and matches whatever is live: nothing now.
        let reply = harness
            .handle
            .forward_control_req(
                AbortReq::new(Rid::new().to_string(), true)
                    .encode()
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!rmp_serde::from_slice::<bool>(&reply).unwrap());
    }

    /// The DP rank 0 listener relays every abort to every peer as the
    /// scheduler's `AbortReq` for the client's rid, built once per peer, and
    /// survives a peer it cannot reach or one that does not answer within
    /// the timeout; a no-op abort is not relayed, the local rank is not
    /// aborted again, and another rank's listener relays nothing.
    #[tokio::test(start_paused = true)]
    async fn abort_peers_relays_the_abort_to_every_peer_from_dp_rank_0() {
        use crate::api_server::core::fanout::testing::DummyTarget;
        use crate::api_server::core::fanout::{ControlTarget, RankId};
        use crate::message::config::DeploymentInfo;
        use futures::future::BoxFuture;
        use std::sync::{Arc, Mutex};

        struct Recording(Vec<RankId>, Arc<Mutex<Vec<ControlRequest>>>);
        impl ControlTarget for Recording {
            fn ranks(&self) -> &[RankId] {
                &self.0
            }
            fn control<'a>(
                &'a self,
                build: &'a RequestBuilder<'a>,
            ) -> BoxFuture<'a, Vec<RankReply>> {
                Box::pin(async move {
                    self.1.lock().unwrap().push(build());
                    vec![RankReply {
                        rank: self.0[0],
                        outcome: Ok(rmp_serde::to_vec(&true).unwrap().into()),
                    }]
                })
            }
        }
        let handle_at = |dp_rank: u32, peer_endpoints: Vec<Arc<dyn ControlTarget>>| {
            let (intake_tx, intake_rx) = flume::unbounded();
            let (abort_tx, abort_rx) = flume::unbounded();
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
                        deployment_info: DeploymentInfo {
                            dp_rank: Some(dp_rank),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    peer_endpoints,
                },
            );
            (handle, intake_rx, abort_rx)
        };

        let seen = Arc::new(Mutex::new(Vec::new()));
        let slow = DummyTarget::delayed(&[3], 10 * PEER_ABORT_TIMEOUT, |_| {
            Ok(rmp_serde::to_vec(&true).unwrap().into())
        });
        let (handle, intake_rx, abort_rx) = handle_at(
            0,
            vec![
                Arc::new(Recording(vec![1], seen.clone())),
                DummyTarget::replying(&[2], |_| Err(CoreError::Unavailable)),
                slow,
            ],
        );
        let started = tokio::time::Instant::now();
        handle.abort_peers("client-rid".into(), false).await;
        assert_eq!(
            started.elapsed(),
            PEER_ABORT_TIMEOUT,
            "a peer that never answers holds the relay for the timeout only"
        );
        handle.abort_peers(String::new(), true).await;
        handle.abort_peers(String::new(), false).await;
        {
            let seen = seen.lock().unwrap();
            let aborts: Vec<(&str, bool)> = seen
                .iter()
                .map(|request| match request {
                    ControlRequest::AbortReq(abort) => (abort.get_rid(), abort.abort_all()),
                    other => panic!("relayed {other:?}"),
                })
                .collect();
            assert_eq!(aborts.len(), 2, "{aborts:?}");
            assert_eq!(aborts[0], ("client-rid", false));
            assert!(aborts[1].1, "{aborts:?}");
            assert_eq!(aborts[1].0.len(), 32, "abort_all rides a minted rid");
        }
        assert!(intake_rx.try_recv().is_err() && abort_rx.try_recv().is_err());

        // Another rank's listener serves its own abort and relays nothing.
        let seen_by_rank_1 = Arc::new(Mutex::new(Vec::new()));
        let (handle, _intake_rx, _abort_rx) = handle_at(
            1,
            vec![Arc::new(Recording(vec![0], seen_by_rank_1.clone()))],
        );
        handle.abort_peers("client-rid".into(), false).await;
        handle.abort_peers(String::new(), true).await;
        assert!(seen_by_rank_1.lock().unwrap().is_empty());

        // At most `PEER_ABORT_MAX_IN_FLIGHT` posts run at once: one more
        // peer than that, each answering after a second, takes two.
        let second = std::time::Duration::from_secs(1);
        let peers = (0..=PEER_ABORT_MAX_IN_FLIGHT as RankId + 1)
            .filter(|&rank| rank != 0)
            .map(|rank| {
                DummyTarget::delayed(&[rank], second, |_| {
                    Ok(rmp_serde::to_vec(&true).unwrap().into())
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(peers.len(), PEER_ABORT_MAX_IN_FLIGHT + 1);
        let (handle, _intake_rx, _abort_rx) = handle_at(0, peers);
        let started = tokio::time::Instant::now();
        handle.abort_peers("client-rid".into(), false).await;
        assert_eq!(started.elapsed(), 2 * second);
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
