//! The transport-neutral deep-health probe shared by every transport's
//! health endpoint.

use crate::api_server::core::error::ApiError;
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::state::CoreState;
use crate::api_server::core::submit::submit;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RequestKind};
use crate::message::sampling::SamplingParams;

/// Sentinel host that makes the KV connector no-op. Parity with
/// `sglang.srt.disaggregation.utils.FAKE_BOOTSTRAP_HOST`.
const FAKE_BOOTSTRAP_HOST: &str = "2.2.2.2";

/// The deep-health verdict: the pipeline produced output within the timeout,
/// or stalled (the transport's 503 — distinct from a submit failure, which is
/// a real error with its own status).
pub(crate) enum HealthStatus {
    Alive,
    Stalled,
}

/// Deep health: confirm the scheduler → detok path is producing output.
///
/// Fires a pre-tokenized 1-token probe (`input_ids = [0]`, skips the tokenizer) so
/// an idle pipeline produces a frame, then watches the *global*
/// [`CoreState::response_activity`] counter (not the probe's own rid) — so a busy
/// server passes immediately and a backlog never false-503s (the analogue of
/// Python's `last_receive_tstamp`).
pub(crate) async fn health_probe(
    state: &CoreState,
    timeout: std::time::Duration,
) -> Result<HealthStatus, ApiError> {
    let baseline = state
        .response_activity
        .load(std::sync::atomic::Ordering::Relaxed);

    // Fire the probe (the heartbeat is the signal, not its own response). A busy
    // scheduler skips it with no terminal frame, so its detok registration is
    // cleaned up only by the `AbortGuard` below.
    //
    // On a PD node the scheduler 400-aborts room-less requests, so inject the
    // same fake bootstrap pair Python uses (`FAKE_BOOTSTRAP_HOST` / room 0).
    let pd = state.server_args.is_disaggregation();
    let probe = GenerateRequest {
        // The `HEALTH_CHECK_<uuid>` rid form
        rid: Rid::new_health_check(),
        input_ids: Some(vec![0]),
        // One greedy token: the cheapest round-trip that still produces a frame.
        sampling_params: SamplingParams {
            max_new_tokens: Some(1),
            temperature: 0.0,
            ..Default::default()
        },
        stream: false,
        bootstrap_host: pd.then(|| FAKE_BOOTSTRAP_HOST.into()),
        bootstrap_room: pd.then_some(0),
        ..Default::default()
    };
    // Hold the receiver so the probe's sink stays open until it completes.
    let (rid, _keepalive) = submit(state, RequestKind::Generate(Box::new(probe))).await?;
    // Deregister on drop (never disarmed): a busy-skipped probe has no terminal
    // frame, so without this abort it leaks one detok entry per call.
    let _abort_guard = AbortGuard::new(state.senders.clone(), rid);

    // Watch the heartbeat advance (timeout frozen at router build, default 20s).
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if state
            .response_activity
            .load(std::sync::atomic::Ordering::Relaxed)
            != baseline
        {
            return Ok(HealthStatus::Alive);
        }
        if tokio::time::Instant::now() >= deadline {
            return Ok(HealthStatus::Stalled);
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}
