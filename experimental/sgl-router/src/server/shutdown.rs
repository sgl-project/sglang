//! Graceful-termination helpers.
//!
//! On SIGTERM we want the EndpointSlice controller to deregister this pod
//! before the HTTP server stops accepting, so in-flight-capable requests stop
//! arriving on a socket that is about to close. The drain below flips `/readyz`
//! to 503 first, then pauses to let that deregistration propagate.

use crate::server::app_context::AppContext;
use std::future::Future;
use std::time::Duration;

/// Begin a graceful-termination drain: flip `/readyz` to 503 so k8s
/// deregisters this pod from its EndpointSlice, then pause `drain` to let that
/// deregistration propagate before the caller stops accepting connections.
/// A zero `drain` flips readiness and returns at once (no pause). This composes
/// with axum's `with_graceful_shutdown`: once this future resolves, axum stops
/// accepting and drains the already-in-flight requests.
///
/// `expedite` is an escape hatch: if it resolves before the pause elapses (an
/// operator re-sending SIGTERM, or a SIGINT), the drain is cut short so the
/// process is not stuck for the full window. Pass [`std::future::pending`] to
/// never expedite.
pub async fn drain_for_termination(
    ctx: &AppContext,
    drain: Duration,
    expedite: impl Future<Output = ()>,
) {
    ctx.mark_not_ready();
    if !drain.is_zero() {
        tracing::info!(
            drain_secs = drain.as_secs(),
            "draining: /readyz now 503, waiting before the server stops accepting"
        );
        tokio::select! {
            _ = tokio::time::sleep(drain) => {}
            _ = expedite => {
                tracing::info!("drain expedited by a second termination signal");
            }
        }
    }
}

/// Await `fut` for at most `dur`, logging a warning if it does not complete in
/// time. Returns `true` if the future completed, `false` on timeout. Used to
/// keep a hung background-task join from silently degrading a graceful
/// shutdown into a SIGKILL with no explanation in the logs.
pub async fn await_with_timeout(fut: impl Future<Output = ()>, dur: Duration, what: &str) -> bool {
    if tokio::time::timeout(dur, fut).await.is_err() {
        tracing::warn!(
            what,
            timeout_secs = dur.as_secs(),
            "shutdown step did not finish in time; proceeding to exit"
        );
        false
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::app_context::AppContext;
    use std::time::Duration;

    #[tokio::test]
    async fn drain_zero_returns_immediately_and_flips_readiness() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        // A zero drain must skip the sleep entirely — guard against a hang.
        let done = tokio::time::timeout(
            Duration::from_secs(1),
            drain_for_termination(&ctx, Duration::ZERO, std::future::pending::<()>()),
        )
        .await;
        assert!(
            done.is_ok(),
            "zero drain must return immediately, not sleep"
        );
        assert!(!ctx.is_ready(), "zero drain still flips readiness off");
    }

    #[tokio::test(start_paused = true)]
    async fn drain_holds_for_the_delay_after_flipping_readiness() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        // A 30 s drain must not have returned within a 10 ms window...
        let returned = tokio::time::timeout(
            Duration::from_millis(10),
            drain_for_termination(&ctx, Duration::from_secs(30), std::future::pending::<()>()),
        )
        .await;
        assert!(
            returned.is_err(),
            "drain must still be sleeping out the configured delay",
        );
        // ...but readiness flipped off on entry, before the sleep.
        assert!(
            !ctx.is_ready(),
            "readiness must flip off before the drain delay elapses",
        );
    }

    #[tokio::test(start_paused = true)]
    async fn drain_is_cut_short_when_expedite_resolves() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        // A second termination signal (here: an already-resolved expedite
        // future) must cut the drain short so an operator re-sending SIGTERM
        // is not stuck for the full window.
        let done = tokio::time::timeout(
            Duration::from_millis(10),
            drain_for_termination(&ctx, Duration::from_secs(3600), std::future::ready(())),
        )
        .await;
        assert!(
            done.is_ok(),
            "an expedite signal must cut the drain short, not wait out 3600 s"
        );
        assert!(!ctx.is_ready(), "readiness still flipped off");
    }

    #[tokio::test(start_paused = true)]
    async fn await_with_timeout_returns_true_when_the_future_completes() {
        let completed =
            await_with_timeout(std::future::ready(()), Duration::from_secs(5), "unit").await;
        assert!(completed, "a ready future must report completion");
    }

    #[tokio::test(start_paused = true)]
    async fn await_with_timeout_returns_false_when_the_future_hangs() {
        let completed =
            await_with_timeout(std::future::pending::<()>(), Duration::from_secs(5), "unit").await;
        assert!(!completed, "a hung future must report the timeout elapsed");
    }
}
