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
        // Checked first, as Python checks `gracefully_exit`: a draining
        // server sends no probe.
        if self.is_draining() {
            return Ok(HealthStatus::Draining);
        }
        if !self.is_serving() {
            return Ok(HealthStatus::NotReady);
        }
        // Detached, as Python's handler runs to completion after its client
        // disconnects: a prober that gives up before `timeout` must not
        // cancel the outcome the drain reads (Python's `UnHealthy`).
        let core = self.clone();
        tokio::spawn(async move { core.run_health_probe(timeout).await })
            .await
            .map_err(|error| CoreError::Internal(format!("health probe task: {error}")))?
    }

    async fn run_health_probe(&self, timeout: Duration) -> Result<HealthStatus, CoreError> {
        let deadline = tokio::time::Instant::now() + timeout;
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
        // A wedged scheduler can leave the intake full: the submission counts
        // against the same deadline, and the heartbeat still decides, as
        // Python judges only `last_receive_tstamp`.
        let _probe =
            match tokio::time::timeout_at(deadline, self.generate_internal(probe, false)).await {
                Ok(probe) => Some(probe?),
                Err(_) => None,
            };

        loop {
            if self.response_activity() != baseline {
                self.inner.drain.set_unhealthy(false);
                return Ok(HealthStatus::Healthy);
            }
            if tokio::time::Instant::now() >= deadline {
                // Python's `UnHealthy`: a drain begun now skips the wait.
                self.inner.drain.set_unhealthy(true);
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
    use crate::api_server::core::DrainEnd;
    use crate::api_server::handlers::AbortEnding::CancelledError;
    use crate::api_server::handlers::test_utils::*;
    use crate::message::request::RequestKind;
    use crate::message::response::{ChunkEvent, ResponseItem, frame_health_ack};
    use crate::runtime::Runnable;
    use crate::tokenizer_manager::from_scheduler::{ActivityCounter, Dispatcher};
    use crate::tokenizer_manager::wiring::{AbortSource, Senders, TmEvent};

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

    /// The drain waits for client calls but not for health probes, and the
    /// last probe's outcome decides whether it waits at all: a stall (Python's
    /// `UnHealthy`) skips the wait, a later healthy probe restores it. Once
    /// draining, health answers without probing the scheduler.
    #[tokio::test]
    async fn drain_tracks_client_calls_and_the_last_probe_outcome() {
        async fn settles<F: std::future::Future>(future: F) -> Option<F::Output> {
            tokio::time::timeout(Duration::from_millis(100), future)
                .await
                .ok()
        }
        async fn probe(
            handle: &CoreHandle,
            intake_rx: &flume::Receiver<TmEvent>,
            activity: Option<&ActivityCounter>,
        ) -> HealthStatus {
            let task = tokio::spawn({
                let handle = handle.clone();
                async move {
                    handle
                        .probe_health(Duration::from_millis(50))
                        .await
                        .unwrap()
                }
            });
            let _probe = accept_intake(intake_rx.recv_async().await.unwrap());
            if let Some(activity) = activity {
                activity.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            task.await.unwrap()
        }

        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let activity: ActivityCounter = Default::default();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            activity.clone(),
            true,
            false,
            Vec::new(),
        );
        let call = handle
            .generate(GenerateRequest {
                rid: "client".into(),
                input_ids: Some(vec![1]),
                ..Default::default()
            })
            .await
            .unwrap();
        let _client = accept_intake(intake_rx.recv_async().await.unwrap());

        assert_eq!(
            probe(&handle, &intake_rx, None).await,
            HealthStatus::Stalled
        );
        assert_eq!(
            probe(&handle, &intake_rx, Some(&activity)).await,
            HealthStatus::Healthy
        );
        handle.begin_drain(false);
        assert_eq!(
            settles(handle.drain_ended()).await,
            None,
            "a client call is in flight"
        );
        assert_eq!(
            handle
                .probe_health(Duration::from_millis(50))
                .await
                .unwrap(),
            HealthStatus::Draining
        );
        assert!(
            intake_rx.try_recv().is_err(),
            "a draining server sends no probe"
        );
        drop(call);
        assert_eq!(settles(handle.drain_ended()).await, Some(DrainEnd::Drained));

        // A stalled last probe: the drain skips the in-flight call.
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            true,
            false,
            Vec::new(),
        );
        let _call = handle
            .generate(GenerateRequest {
                rid: "client".into(),
                input_ids: Some(vec![1]),
                ..Default::default()
            })
            .await
            .unwrap();
        let _client = accept_intake(intake_rx.recv_async().await.unwrap());
        assert_eq!(
            probe(&handle, &intake_rx, None).await,
            HealthStatus::Stalled
        );
        handle.begin_drain(false);
        assert_eq!(settles(handle.drain_ended()).await, Some(DrainEnd::Forced));
    }

    /// A router's probe client gives up long before the probe's own timeout;
    /// the stall must still be recorded for the drain (Python's handler also
    /// runs on after its client left). A wedged scheduler can leave the intake
    /// full: that counts against the same deadline instead of hanging, and the
    /// heartbeat still decides the outcome.
    #[tokio::test]
    async fn probe_outcome_survives_the_client_and_a_full_intake() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            Default::default(),
            true,
            false,
            Vec::new(),
        );
        let _call = handle
            .generate(GenerateRequest {
                rid: "client".into(),
                input_ids: Some(vec![1]),
                ..Default::default()
            })
            .await
            .unwrap();
        let _client = accept_intake(intake_rx.recv_async().await.unwrap());
        // The client disconnects after 20 ms; the probe's timeout is 100 ms.
        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                handle.probe_health(Duration::from_millis(100))
            )
            .await
            .is_err()
        );
        let _probe = accept_intake(intake_rx.recv_async().await.unwrap());
        tokio::time::sleep(Duration::from_millis(300)).await;
        handle.begin_drain(false);
        assert_eq!(
            tokio::time::timeout(Duration::from_millis(100), handle.drain_ended())
                .await
                .ok(),
            Some(DrainEnd::Forced)
        );

        // Nobody drains this intake. The heartbeat still decides: responses
        // to calls already admitted keep the server healthy.
        let (intake_tx, _intake_rx) = flume::bounded(0);
        let (abort_tx, _abort_rx) = flume::unbounded();
        let activity: ActivityCounter = Default::default();
        let handle = configured_handle(
            intake_tx,
            abort_tx,
            8,
            activity.clone(),
            true,
            false,
            Vec::new(),
        );
        let probe = || async {
            tokio::time::timeout(
                Duration::from_secs(2),
                handle.probe_health(Duration::from_millis(50)),
            )
            .await
            .expect("the probe hung on a full intake")
            .unwrap()
        };
        assert_eq!(probe().await, HealthStatus::Stalled);
        let heartbeat = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            activity.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert_eq!(probe().await, HealthStatus::Healthy);
        heartbeat.await.unwrap();
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
