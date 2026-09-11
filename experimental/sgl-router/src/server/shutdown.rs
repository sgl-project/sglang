// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Graceful-termination helpers.
//!
//! On SIGTERM the pod is on its way out, but the data plane does not know it
//! yet: kube-proxy keeps routing here until the endpoint removal propagates to
//! it (an external load balancer deregisters on its own probe cadence
//! instead — see [`drain_for_termination`]). Requests sent in that window
//! reach a socket that is about to stop accepting and fail at the client. The
//! drain below holds the listener open for a fixed, operator-set window sized
//! to cover that propagation, with `/readyz` already reporting 503.

use crate::server::app_context::AppContext;
use std::future::Future;
use std::time::Duration;

/// Begin a graceful-termination drain: flip `/readyz` to 503, then keep
/// serving for `drain` before the caller stops accepting connections. A zero
/// `drain` flips readiness and returns at once. This composes with axum's
/// `with_graceful_shutdown`: once this future resolves, axum stops accepting
/// and drains the already-in-flight requests.
///
/// The two deregistration mechanisms, and which one the pause is sized for:
///
/// 1. **Endpoint removal.** For a pod deletion or rolling update the
///    EndpointSlice controller marks this pod's endpoint not-ready the moment
///    the `deletionTimestamp` is stamped — it does not wait on a probe. The
///    pause covers the propagation of that to kube-proxy on every node. This
///    is the mechanism the default drain is sized for.
/// 2. **The `/readyz` flip.** Probe-driven deregistration (an external load
///    balancer, or a kubelet readiness probe on a pod that is not being
///    deleted) needs `failureThreshold` consecutive failures at
///    `periodSeconds` apart before it acts. A drain shorter than that product
///    never gets observed, so operators relying on this path must raise the
///    drain to match their own probe cadence — the default does not do it for
///    them.
///
/// `expedite` cuts the *pause* short: if it resolves first (a further
/// termination signal), the drain returns early so the process is not held for
/// a window that has stopped being useful. It does not reach the axum
/// in-flight drain that runs afterwards, which is unbounded. Pass
/// [`std::future::pending`] to never expedite. Note that two signals delivered
/// close enough together coalesce into one notification, so a drain already
/// running takes one *further* signal to expedite.
pub async fn drain_for_termination(
    ctx: &AppContext,
    drain: Duration,
    expedite: impl Future<Output = ()>,
) {
    ctx.mark_not_ready();
    if drain.is_zero() {
        // Still say so: without this line `--shutdown-drain-secs 0` produces a
        // shutdown log indistinguishable from an image that predates the drain.
        tracing::info!("/readyz now 503; drain pause disabled (shutdown_drain_secs=0)");
        return;
    }
    tracing::info!(
        drain_secs = drain.as_secs(),
        "draining: /readyz now 503, waiting before the server stops accepting"
    );
    tokio::select! {
        _ = tokio::time::sleep(drain) => {
            tracing::info!(
                drain_secs = drain.as_secs(),
                "drain pause elapsed; the server now stops accepting",
            );
        }
        _ = expedite => {
            tracing::info!("drain pause expedited by a further termination signal");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn drain_zero_flips_readiness_without_pausing() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        // Real time, and an elapsed-time assertion rather than a `timeout`: a
        // "minimum safe drain" floor added to the production path would still
        // fit inside any timeout generous enough not to be flaky, so only
        // measuring the elapsed time actually pins "0 means no pause".
        let started = std::time::Instant::now();
        drain_for_termination(&ctx, Duration::ZERO, std::future::pending::<()>()).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(50),
            "zero drain must not pause, slept {elapsed:?}",
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

    /// The pause must last *the configured* time — not merely "some time".
    /// Two distinct values, because a hardcoded constant substituted for
    /// `drain` satisfies any single-value test.
    #[tokio::test(start_paused = true)]
    async fn drain_holds_for_exactly_the_configured_delay() {
        for secs in [5_u64, 30] {
            let ctx = Arc::new(AppContext::stub());
            ctx.mark_ready();
            let drain_ctx = Arc::clone(&ctx);
            let handle = tokio::spawn(async move {
                drain_for_termination(
                    &drain_ctx,
                    Duration::from_secs(secs),
                    std::future::pending::<()>(),
                )
                .await;
            });
            // Let the task register its sleep before advancing: a timer that
            // has not been created yet cannot be advanced past.
            tokio::task::yield_now().await;

            // One millisecond shy of the deadline the drain must still be held.
            tokio::time::advance(Duration::from_millis(secs * 1000 - 1)).await;
            tokio::task::yield_now().await;
            assert!(
                !handle.is_finished(),
                "a {secs} s drain returned before its configured delay elapsed",
            );

            // Past it, it must return promptly.
            tokio::time::advance(Duration::from_millis(2)).await;
            tokio::time::timeout(Duration::from_secs(1), handle)
                .await
                .unwrap_or_else(|_| panic!("a {secs} s drain must return once its delay elapses"))
                .expect("drain task joined cleanly");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn drain_is_cut_short_when_expedite_resolves() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        // A further termination signal (here: an already-resolved expedite
        // future) must cut the pause short so an operator re-sending SIGTERM
        // is not held for the full window.
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
}
