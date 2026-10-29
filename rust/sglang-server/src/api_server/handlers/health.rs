//! The deep health probe and the startup warmup, both one internal generation.

use std::time::Duration;

use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;
use crate::message::sampling::SamplingParams;

use super::handle::CoreHandle;
use crate::api_server::core::{CoreError, CoreEvent, HealthStatus};

/// Sentinel host that makes the KV connector no-op. Parity with
/// `sglang.srt.disaggregation.utils.FAKE_BOOTSTRAP_HOST`.
const FAKE_BOOTSTRAP_HOST: &str = "2.2.2.2";

impl CoreHandle {
    /// Confirm that scheduler output is moving. The response heartbeat is the
    /// signal rather than this probe's own result, matching Python's
    /// `last_receive_tstamp` behavior on a busy server.
    pub(crate) async fn probe_health(&self, timeout: Duration) -> Result<HealthStatus, CoreError> {
        if !self.is_serving() {
            return Ok(HealthStatus::NotReady);
        }

        let baseline = self.response_activity();
        let probe = GenerateRequest {
            rid: Rid::new_health_check(),
            input_ids: Some(vec![0]),
            sampling_params: SamplingParams {
                max_new_tokens: Some(1),
                temperature: 0.0,
                ..Default::default()
            },
            stream: false,
            bootstrap_host: self
                .inner
                .is_disaggregation
                .then(|| FAKE_BOOTSTRAP_HOST.into()),
            bootstrap_room: self.inner.is_disaggregation.then_some(0),
            ..Default::default()
        };
        // The probe is intentionally not drained. Its call stays alive while
        // the heartbeat is observed, then aborts on drop to clean up a probe
        // that a busy scheduler skipped without producing a terminal frame.
        let _probe = self.generate_internal(probe, false).await?;

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.response_activity() != baseline {
                return Ok(HealthStatus::Healthy);
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(HealthStatus::Stalled);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// One generation mirroring the Python `_execute_server_warmup` text request;
    /// VLMs warm only the text path. The caller decides readiness.
    pub(crate) async fn warm_up(&self, skip_tokenizer_init: bool) -> Result<(), CoreError> {
        let (text, input_ids) = if skip_tokenizer_init {
            (None, Some(vec![10, 11, 12]))
        } else {
            (Some("The capital city of France is".to_string()), None)
        };
        let mut call = self
            .generate_internal(
                GenerateRequest {
                    rid: Rid::new(),
                    text,
                    input_ids,
                    sampling_params: SamplingParams {
                        max_new_tokens: Some(8),
                        temperature: 0.0,
                        ..Default::default()
                    },
                    stream: false,
                    bootstrap_host: self
                        .inner
                        .is_disaggregation
                        .then(|| FAKE_BOOTSTRAP_HOST.into()),
                    bootstrap_room: self.inner.is_disaggregation.then_some(0),
                    ..Default::default()
                },
                true,
            )
            .await?;
        loop {
            match call.recv().await {
                Some(CoreEvent::Finished(_)) => return Ok(()),
                Some(CoreEvent::Failed(e)) => return Err(e),
                Some(CoreEvent::Delta(_)) => {}
                None => return Err(CoreError::ResponseTruncated),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::AbortEnding::CancelledError;
    use crate::api_server::handlers::test_utils::*;
    use crate::message::request::RequestKind;
    use crate::message::response::{ChunkEvent, ResponseItem, frame_health_ack};
    use crate::runtime::Runnable;
    use crate::tokenizer_manager::from_scheduler::{ActivityCounter, Dispatcher};
    use crate::tokenizer_manager::wiring::{AbortSource, Senders};

    /// A busy scheduler skips the probe and acknowledges it after its next step.
    /// The dispatcher counts the acknowledgement as scheduler output, so the
    /// probe passes without any batch frame.
    #[tokio::test]
    async fn skipped_probe_passes_on_health_ack_and_cleans_up() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let activity: ActivityCounter = Default::default();
        let (from_scheduler_tx, from_scheduler_rx) =
            crate::tokenizer_manager::channel::from_scheduler(8);
        let (detok_tx, detok_rx) = flume::unbounded();
        let (_shutdown_tx, shutdown_rx) = flume::bounded(1);
        let dispatcher = Dispatcher::new(
            from_scheduler_rx,
            Senders {
                tok_manager_tx: flume::unbounded().0,
                abort_tx: flume::unbounded().0,
                tokenizer_tx: flume::unbounded().0,
                detokenizer_tx: vec![detok_tx],
            },
            activity.clone(),
            shutdown_rx,
        );
        let dispatcher = std::thread::spawn(move || dispatcher.run());
        let handle = configured_handle(intake_tx, abort_tx, 8, activity, true, false, Vec::new());
        let probe_handle = handle.clone();
        let probe =
            tokio::spawn(async move { probe_handle.probe_health(Duration::from_secs(1)).await });

        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        let rid = request.rid.clone();
        assert!(rid.as_str().starts_with("HEALTH_CHECK_"));
        assert!(
            !handle.abort(None, CancelledError),
            "internal probes must not enter the public abort-all registry"
        );
        assert!(from_scheduler_tx.push(frame_health_ack()));

        assert_eq!(probe.await.unwrap().unwrap(), HealthStatus::Healthy);
        assert!(matches!(
            abort_rx.recv().unwrap(),
            AbortSource::Guard(aborted) if aborted == rid
        ));
        assert!(abort_rx.try_recv().is_err());
        drop(from_scheduler_tx);
        dispatcher.join().unwrap();
        assert!(
            detok_rx.try_recv().is_err(),
            "an acknowledgement routes nothing"
        );
    }

    #[tokio::test]
    async fn stalled_pd_probe_has_bootstrap_pair_and_cleans_up() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            true,
            true,
            Vec::new(),
        );

        let probe_task =
            tokio::spawn(
                async move { handle.probe_health(Duration::from_millis(1)).await.unwrap() },
            );
        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        let rid = request.rid.clone();
        let RequestKind::Generate(probe) = request.kind else {
            panic!("health must submit a generation request");
        };
        assert_eq!(probe.bootstrap_host.as_deref(), Some(FAKE_BOOTSTRAP_HOST));
        assert_eq!(probe.bootstrap_room, Some(0));
        assert_eq!(probe_task.await.unwrap(), HealthStatus::Stalled);
        assert!(matches!(
            abort_rx.recv().unwrap(),
            AbortSource::Guard(aborted) if aborted == rid
        ));
    }

    #[tokio::test]
    async fn warm_up_succeeds_on_finished_generation() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let warm_up = tokio::spawn(async move { handle.warm_up(false).await });

        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        let RequestKind::Generate(warmup) = &request.kind else {
            panic!("warmup must submit a generation request");
        };
        assert_eq!(
            warmup.text.as_deref(),
            Some("The capital city of France is")
        );
        assert_eq!(warmup.input_ids, None);
        assert_eq!(warmup.sampling_params.max_new_tokens, Some(8));
        assert_eq!(warmup.bootstrap_host, None);
        assert!(
            !harness.handle.abort(None, CancelledError),
            "startup warmup must not enter the public abort-all registry"
        );
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent::default()))
            .unwrap();

        assert!(warm_up.await.unwrap().is_ok());
        assert!(!harness.handle.is_ready());
    }

    #[tokio::test]
    async fn pd_warm_up_without_tokenizer_fails_on_truncated_stream() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            false,
            true,
            Vec::new(),
        );
        let warm_up = tokio::spawn(async move { handle.warm_up(true).await });

        let request = accept_intake(intake_rx.recv_async().await.unwrap());
        let RequestKind::Generate(warmup) = &request.kind else {
            panic!("warmup must submit a generation request");
        };
        assert_eq!(warmup.text, None);
        assert_eq!(warmup.input_ids, Some(vec![10, 11, 12]));
        assert_eq!(warmup.bootstrap_host.as_deref(), Some(FAKE_BOOTSTRAP_HOST));
        assert_eq!(warmup.bootstrap_room, Some(0));
        drop(request);

        assert!(warm_up.await.unwrap().is_err());
    }
}
