//! Process-local opt-in cancellation retirement waits for api.v1 Abort.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::watch;
use tokio::time::Instant;

use super::generate::{CaptureError, CapturedGeneration};
use super::handle::{CoreHandle, translate_runtime_error};
use crate::api_server::core::CoreError;
use crate::message::ids::Rid;
use crate::message::io_struct::{
    CancellationRetirementOutput, CancellationRetirementReq, CancellationRetirementStatus,
    ControlRequest,
};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;

const CAPACITY: usize = 256;
const RETENTION: Duration = Duration::from_secs(300);
pub(super) const ENV_ENABLE: &str = "SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS";

/// Narrow outcome for an opt-in retirement wait.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AbortWaitResult {
    pub(crate) success: bool,
    pub(crate) retirement_confirmed: bool,
}

type SharedOutcome = Arc<watch::Sender<Option<Result<AbortWaitResult, CoreError>>>>;

struct Operation {
    public_rid: String,
    /// Original admission time; waiting clients must not extend this deadline.
    created: Instant,
    outcome: SharedOutcome,
}

/// Bounded capture/replay registry owned by [`super::handle::CoreInner`].
pub(super) struct RetirementRegistry {
    ops: Mutex<HashMap<String, Operation>>,
    /// Snapshotted once at core construction (`environ::env_bool`); tests may
    /// flip only their own core via [`CoreHandle::set_cancel_retirement_receipts_enabled`].
    enabled: AtomicBool,
}

impl RetirementRegistry {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            ops: Mutex::new(HashMap::new()),
            enabled: AtomicBool::new(enabled),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    fn expire_locked(ops: &mut HashMap<String, Operation>, now: Instant) {
        ops.retain(|_, op| {
            if now.saturating_duration_since(op.created) < RETENTION {
                return true;
            }
            // A waiter may still hold this channel when a retry prunes the
            // registry before the background timer is polled.
            op.outcome.send_if_modified(|value| {
                if value.is_some() {
                    return false;
                }
                *value = Some(Err(retirement_expired()));
                true
            });
            false
        });
    }
}

impl Default for RetirementRegistry {
    fn default() -> Self {
        Self::new(false)
    }
}

impl CoreHandle {
    /// Opt-in exact-request retirement wait. Validates configuration and
    /// capture before any cancel. Shared operations survive RPC disconnect.
    pub(crate) async fn abort_and_wait(
        &self,
        public_rid: &str,
        cancellation_id: &str,
        wait_ms: u32,
    ) -> Result<AbortWaitResult, CoreError> {
        if public_rid.is_empty() {
            return Err(CoreError::InvalidArgument(
                "wait_for_retirement_ms requires an exact nonempty rid".into(),
            ));
        }
        if cancellation_id.is_empty() || cancellation_id.len() > 128 {
            return Err(CoreError::InvalidArgument(
                "wait_for_retirement_ms requires a nonempty cancellation_id of at most 128 UTF-8 bytes"
                    .into(),
            ));
        }
        if wait_ms > 30_000 {
            return Err(CoreError::InvalidArgument(
                "wait_for_retirement_ms must be <= 30000".into(),
            ));
        }
        if let Some(reason) = self.retirement_scope_error() {
            return Err(CoreError::Unimplemented(reason));
        }
        if !self.inner.retirements.enabled() {
            return Err(CoreError::Unimplemented(
                "cancellation retirement receipts require SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS=1"
                    .into(),
            ));
        }

        let rx = {
            let mut ops = self.inner.retirements.ops.lock().unwrap();
            RetirementRegistry::expire_locked(&mut ops, Instant::now());
            if let Some(existing) = ops.get(cancellation_id) {
                if existing.public_rid != public_rid {
                    return Err(CoreError::FailedPrecondition(
                        "cancellation_id is already bound to another request".into(),
                    ));
                }
                existing.outcome.subscribe()
            } else {
                match self.inner.generations.capture_one(public_rid) {
                    Ok(captured) => {
                        if ops.len() >= CAPACITY {
                            return Err(CoreError::Overloaded(
                                "cancellation receipt capacity reached; retry later".into(),
                            ));
                        }
                        let (tx, rx) = watch::channel(None);
                        let outcome = Arc::new(tx);
                        let created = Instant::now();
                        // Server-minted id, distinct from the client key and from Python
                        // HTTP receipt operations sharing the scheduler registry.
                        let scheduler_cancellation_id = format!("rust-{}", Rid::new().as_str());
                        let request_instance_id = captured.rid.as_str().to_owned();
                        ops.insert(
                            cancellation_id.to_owned(),
                            Operation {
                                public_rid: public_rid.to_owned(),
                                created,
                                outcome: outcome.clone(),
                            },
                        );
                        drop(ops);
                        self.spawn_retirement(
                            captured,
                            scheduler_cancellation_id,
                            request_instance_id,
                            created,
                            outcome,
                        );
                        rx
                    }
                    Err(capture_err) => {
                        if ops.len() >= CAPACITY {
                            return Err(CoreError::Overloaded(
                                "cancellation receipt capacity reached; retry later".into(),
                            ));
                        }
                        let err = match capture_err {
                            CaptureError::NotFound => CoreError::FailedPrecondition(
                                "cancellation target is unknown or already finished".into(),
                            ),
                            CaptureError::Ambiguous => CoreError::FailedPrecondition(
                                "wait_for_retirement_ms refuses ambiguous duplicate public rids"
                                    .into(),
                            ),
                        };
                        let (tx, rx) = watch::channel(None);
                        let outcome = Arc::new(tx);
                        ops.insert(
                            cancellation_id.to_owned(),
                            Operation {
                                public_rid: public_rid.to_owned(),
                                created: Instant::now(),
                                outcome: outcome.clone(),
                            },
                        );
                        // Retain the failed capture so the same key never selects
                        // later work under a reused public RID within the window.
                        outcome.send_replace(Some(Err(err)));
                        rx
                    }
                }
            }
        };

        self.wait_retirement(rx, wait_ms).await
    }

    /// Test-only: enable/disable receipts for this core without mutating process env.
    #[cfg(test)]
    pub(crate) fn set_cancel_retirement_receipts_enabled(&self, enabled: bool) {
        self.inner
            .retirements
            .enabled
            .store(enabled, Ordering::Release);
    }

    /// Test-only: whether the shared outcome has been published (no subscribe).
    #[cfg(test)]
    pub(crate) fn retirement_outcome_published(&self, cancellation_id: &str) -> bool {
        self.retirement_outcome(cancellation_id).is_some()
    }

    /// Test-only: published outcome without expiring or subscribing.
    #[cfg(test)]
    pub(crate) fn retirement_outcome(
        &self,
        cancellation_id: &str,
    ) -> Option<Result<AbortWaitResult, CoreError>> {
        let ops = self.inner.retirements.ops.lock().unwrap();
        ops.get(cancellation_id)
            .and_then(|op| op.outcome.borrow().clone())
    }

    /// Test-only: live watch receivers for an operation (proves zero-receiver publish).
    #[cfg(test)]
    pub(crate) fn retirement_receiver_count(&self, cancellation_id: &str) -> usize {
        let ops = self.inner.retirements.ops.lock().unwrap();
        ops.get(cancellation_id)
            .map(|op| op.outcome.receiver_count())
            .unwrap_or(0)
    }

    fn retirement_scope_error(&self) -> Option<String> {
        if self.inner.multimodal {
            return Some(
                "cancellation retirement receipts do not support multimodal models".into(),
            );
        }
        if self.inner.is_disaggregation {
            return Some(
                "cancellation retirement receipts do not support disaggregated serving".into(),
            );
        }
        if self.inner.metadata.deployment_info.dp_size > 1 {
            return Some(
                "cancellation retirement receipts do not support distributed DP serving".into(),
            );
        }
        if !self.inner.peer_endpoints.is_empty() {
            return Some(
                "cancellation retirement receipts do not support multi-rank peer fan-out".into(),
            );
        }
        None
    }

    fn spawn_retirement(
        &self,
        captured: CapturedGeneration,
        scheduler_cancellation_id: String,
        request_instance_id: String,
        created: Instant,
        outcome: SharedOutcome,
    ) {
        let handle = self.clone();
        tokio::spawn(async move {
            let deadline = created + RETENTION;
            let result = match tokio::time::timeout_at(
                deadline,
                handle.run_retirement(
                    captured,
                    scheduler_cancellation_id,
                    request_instance_id,
                    deadline,
                ),
            )
            .await
            {
                Ok(result) if Instant::now() < deadline => result,
                _ => Err(retirement_expired()),
            };
            // Retain the value with zero receivers, but never overwrite an
            // expiry already published by a pruning client.
            outcome.send_if_modified(|value| {
                if value.is_some() {
                    return false;
                }
                *value = Some(result);
                true
            });
        });
    }

    async fn run_retirement(
        &self,
        captured: CapturedGeneration,
        scheduler_cancellation_id: String,
        request_instance_id: String,
        deadline: Instant,
    ) -> Result<AbortWaitResult, CoreError> {
        if Instant::now() >= deadline {
            return Err(retirement_expired());
        }
        // Pending admission can be fenced without a scheduler receipt: Intake
        // never claimed the request. Native/MM configs are rejected above.
        if self.inner.generations.fence_pending(&captured.rid) {
            let _ = self
                .inner
                .generations
                .notify_retired(&captured.rid, &self.inner.abort_tx);
            return Ok(AbortWaitResult {
                success: true,
                retirement_confirmed: true,
            });
        }

        // Already accepted: do not send legacy Abort first — its priority lane
        // can remove the request before the independent receipt observer runs.
        let correlation = Rid::new();
        let control = ControlRequest::CancellationRetirementReq(CancellationRetirementReq::new(
            correlation.as_str().to_owned(),
            scheduler_cancellation_id.clone(),
            request_instance_id.clone(),
            Some(captured.rid.as_str().to_owned()),
        ));
        let bytes = self.submit_retirement_control(control).await?;
        // timeout_at may poll an already-ready response before its timer.
        // Check explicitly before trusting or acting on a late positive.
        if Instant::now() >= deadline {
            return Err(retirement_expired());
        }
        let output = CancellationRetirementOutput::decode(&bytes)
            .map_err(|e| CoreError::InvalidResponse(e.to_string()))?;
        if output.rid != correlation.as_str()
            || output.cancellation_id != scheduler_cancellation_id
            || output.request_instance_id != request_instance_id
        {
            return Err(CoreError::FailedPrecondition(
                "cancellation retirement receipt identity mismatch".into(),
            ));
        }
        match output.status {
            CancellationRetirementStatus::Retired => {
                let _ = self
                    .inner
                    .generations
                    .notify_retired(&captured.rid, &self.inner.abort_tx);
                Ok(AbortWaitResult {
                    success: true,
                    retirement_confirmed: true,
                })
            }
            CancellationRetirementStatus::Unsupported => {
                Err(CoreError::Unimplemented(if output.reason.is_empty() {
                    "cancellation retirement is unsupported for this configuration".into()
                } else {
                    output.reason
                }))
            }
            CancellationRetirementStatus::Unknown => {
                Err(CoreError::FailedPrecondition(if output.reason.is_empty() {
                    "cancellation retirement is unknown".into()
                } else {
                    output.reason
                }))
            }
        }
    }

    async fn submit_retirement_control(
        &self,
        request: ControlRequest,
    ) -> Result<bytes::Bytes, CoreError> {
        // Keep the control call alive independently of any Generate stream.
        let mut call = self.submit(RequestKind::Control(Box::new(request))).await?;
        match call.recv().await {
            Some(ResponseItem::Control(bytes)) => Ok(bytes),
            Some(ResponseItem::Error(error)) => Err(translate_runtime_error(error)),
            Some(_) => Err(CoreError::InvalidResponse(
                "unexpected generation output for cancellation retirement".into(),
            )),
            None => Err(CoreError::Unavailable),
        }
    }

    async fn wait_retirement(
        &self,
        mut rx: watch::Receiver<Option<Result<AbortWaitResult, CoreError>>>,
        wait_ms: u32,
    ) -> Result<AbortWaitResult, CoreError> {
        if let Some(done) = rx.borrow().clone() {
            return done;
        }
        let deadline = Instant::now() + Duration::from_millis(u64::from(wait_ms));
        loop {
            match tokio::time::timeout_at(deadline, rx.changed()).await {
                Ok(Ok(())) => {
                    if let Some(done) = rx.borrow().clone() {
                        return done;
                    }
                }
                Ok(Err(_)) => return Err(CoreError::Unavailable),
                Err(_) => {
                    return Ok(AbortWaitResult {
                        success: true,
                        retirement_confirmed: false,
                    });
                }
            }
        }
    }
}

fn retirement_expired() -> CoreError {
    CoreError::FailedPrecondition("cancellation retirement receipt expired".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

    fn handle(
        multimodal: bool,
    ) -> (
        CoreHandle,
        flume::Receiver<TmEvent>,
        flume::Receiver<AbortSource>,
    ) {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 2,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::default(),
                metrics: None,
                request_log: None,
                generate_validator: None,
            },
        );
        core.set_cancel_retirement_receipts_enabled(true);
        (core, intake_rx, abort_rx)
    }

    #[tokio::test]
    async fn pending_fence_confirms_retirement_without_scheduler_control() {
        let (core, intake_rx, abort_rx) = handle(false);
        let _pending = core.hold_generation("fence-me");
        let result = core
            .abort_and_wait("fence-me", "op-fence", 1_000)
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.retirement_confirmed);
        assert!(intake_rx.try_recv().is_err());
        // Positive proof: Local cleanup only, never Guard/Request.
        assert!(matches!(
            abort_rx.try_recv(),
            Ok(AbortSource::Local(rid)) if rid.client_facing() == "fence-me"
        ));
        assert!(abort_rx.try_recv().is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn pruning_expires_existing_waiters_before_background_timer_runs() {
        let (tx, rx) = watch::channel(None);
        let outcome = Arc::new(tx);
        let created = Instant::now();
        let mut ops = HashMap::from([(
            "op".into(),
            Operation {
                public_rid: "request".into(),
                created,
                outcome: outcome.clone(),
            },
        )]);
        RetirementRegistry::expire_locked(&mut ops, created + RETENTION);
        assert!(ops.is_empty());
        assert!(matches!(
            rx.borrow().as_ref(),
            Some(Err(CoreError::FailedPrecondition(_)))
        ));
    }

    #[tokio::test]
    async fn same_key_different_public_rid_is_rejected() {
        let (core, _, _) = handle(false);
        let _a = core.hold_generation("a");
        let _b = core.hold_generation("b");
        assert!(
            core.abort_and_wait("a", "shared-key", 1_000)
                .await
                .unwrap()
                .retirement_confirmed
        );
        let err = core
            .abort_and_wait("b", "shared-key", 1_000)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::FailedPrecondition(_)));
    }

    #[tokio::test]
    async fn unknown_target_ambiguous_and_multimodal_fail_closed() {
        let (core, intake_rx, _) = handle(false);
        let err = core
            .abort_and_wait("missing", "op-missing", 0)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::FailedPrecondition(_)));
        assert!(intake_rx.try_recv().is_err());

        let _first = core.hold_generation("dup");
        let _second = core.hold_generation("dup");
        let err = core.abort_and_wait("dup", "op-dup", 0).await.unwrap_err();
        assert!(matches!(err, CoreError::FailedPrecondition(_)));

        let (mm, intake_rx, _) = handle(true);
        let _pending = mm.hold_generation("mm");
        let err = mm.abort_and_wait("mm", "op-mm", 0).await.unwrap_err();
        assert!(matches!(err, CoreError::Unimplemented(_)));
        assert!(intake_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn unknown_capture_is_cached_against_later_rid_reuse() {
        let (core, intake_rx, _) = handle(false);
        let err = core
            .abort_and_wait("reuse", "op-reuse", 0)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::FailedPrecondition(_)));

        let _pending = core.hold_generation("reuse");
        let again = core
            .abort_and_wait("reuse", "op-reuse", 1_000)
            .await
            .unwrap_err();
        assert!(matches!(again, CoreError::FailedPrecondition(_)));
        // Must not start new capture/control work under the cached error key.
        assert!(intake_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn capacity_retains_unexpired_finals_and_rejects_new() {
        let (core, _, _) = handle(false);
        for i in 0..CAPACITY {
            let err = core
                .abort_and_wait(&format!("gone-{i}"), &format!("cap-{i}"), 0)
                .await
                .unwrap_err();
            assert!(matches!(err, CoreError::FailedPrecondition(_)));
        }
        let err = core
            .abort_and_wait("gone-extra", "cap-extra", 0)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::Overloaded(_)));
        // An existing key still replays; it was not evicted for capacity.
        let replay = core.abort_and_wait("gone-0", "cap-0", 0).await.unwrap_err();
        assert!(matches!(replay, CoreError::FailedPrecondition(_)));
    }

    #[tokio::test]
    async fn core_validates_rid_opid_and_wait_bounds() {
        let (core, _, _) = handle(false);
        assert!(matches!(
            core.abort_and_wait("", "op", 0).await.unwrap_err(),
            CoreError::InvalidArgument(_)
        ));
        assert!(matches!(
            core.abort_and_wait("r", "", 0).await.unwrap_err(),
            CoreError::InvalidArgument(_)
        ));
        assert!(matches!(
            core.abort_and_wait("r", &"x".repeat(129), 0)
                .await
                .unwrap_err(),
            CoreError::InvalidArgument(_)
        ));
        assert!(matches!(
            core.abort_and_wait("r", "op", 30_001).await.unwrap_err(),
            CoreError::InvalidArgument(_)
        ));
    }

    #[tokio::test]
    async fn dropped_zero_poll_waiter_still_replays_published_fence() {
        let (core, _, _) = handle(false);
        let _pending = core.hold_generation("keep");
        let _ = core.abort_and_wait("keep", "op-keep", 0).await.unwrap();
        // Explicit barrier: outcome published while no live waiter remains.
        while !core.retirement_outcome_published("op-keep") {
            tokio::task::yield_now().await;
        }
        assert_eq!(core.retirement_receiver_count("op-keep"), 0);
        let confirmed = core.abort_and_wait("keep", "op-keep", 1_000).await.unwrap();
        assert!(confirmed.success);
        assert!(confirmed.retirement_confirmed);
    }
}
