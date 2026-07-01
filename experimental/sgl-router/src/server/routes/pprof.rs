// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! CPU flamegraph endpoint, only compiled with `--features profiling` (see
//! `Cargo.toml`) — never present in the normal production image. Uses
//! `pprof`'s `SIGPROF`/`setitimer` sampling profiler, a pure userspace
//! mechanism, so it works unprivileged in a container regardless of
//! `perf_event_paranoid` — unlike `perf`, which needs `perf_event_open` and
//! is blocked outright on hardened nodes.

use axum::extract::Query;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use std::time::Duration;

#[derive(Deserialize)]
pub struct ProfileParams {
    #[serde(default = "default_seconds")]
    seconds: u64,
    #[serde(default = "default_frequency")]
    frequency: i32,
}

fn default_seconds() -> u64 {
    10
}

fn default_frequency() -> i32 {
    99
}

/// Sampling frequency (Hz) bounds. `pprof`'s `Timer` divides `1_000_000` by
/// the raw frequency to get a `setitimer` interval in microseconds: zero
/// panics (integer division by zero) INSIDE `ProfilerGuardBuilder::build()`,
/// after it has already flipped the process-global profiler to `running` but
/// before the guard exists to `Drop` and reset that flag — one bad request
/// permanently wedges every later `/debug/pprof/profile` call for the rest of
/// the process's life ("profiler is running"). Negative or very large values
/// don't panic but silently disarm the timer (an invalid or zero-length
/// `setitimer` interval), returning an empty flamegraph with no error. Reject
/// all three shapes before they ever reach `pprof`.
const MIN_FREQUENCY_HZ: i32 = 1;
const MAX_FREQUENCY_HZ: i32 = 1000;

/// Upper bound on how long an unauthenticated caller can hold a profiling
/// session (and the handler's task) open. This is a debug-only, feature-gated
/// endpoint, so the risk is limited to whoever can already reach the pod —
/// still cheap insurance against `?seconds=` typos tying up a session for
/// unreasonably long.
const MAX_SECONDS: u64 = 300;

/// `GET /debug/pprof/profile?seconds=10&frequency=99` — samples the whole
/// process (every thread, via the signal handler) for `seconds` and returns a
/// flamegraph SVG. The `.await` on `sleep` yields this task back to the
/// scheduler for the whole window, so its OS thread is free to run other
/// request-handling tasks during that time and the bulk of the sample
/// reflects them, not this handler. The brief report-build/render step right
/// after the sleep still runs while the guard is active, though, and will
/// show up too.
pub async fn profile(Query(params): Query<ProfileParams>) -> Response {
    if !(MIN_FREQUENCY_HZ..=MAX_FREQUENCY_HZ).contains(&params.frequency) {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "frequency must be in {MIN_FREQUENCY_HZ}..={MAX_FREQUENCY_HZ} Hz, got {}",
                params.frequency
            ),
        )
            .into_response();
    }
    if params.seconds == 0 || params.seconds > MAX_SECONDS {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "seconds must be in 1..={MAX_SECONDS}, got {}",
                params.seconds
            ),
        )
            .into_response();
    }

    // `pprof`'s process-wide profiler is a singleton: only one session can
    // run at a time, so a second concurrent request here is an expected
    // collision (two operators debugging at once), not a broken profiler —
    // called out explicitly so the caller knows to retry instead of assuming
    // this endpoint is dead.
    let guard = match pprof::ProfilerGuardBuilder::default()
        .frequency(params.frequency)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
    {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                error = %e,
                seconds = params.seconds,
                frequency = params.frequency,
                "cpu profiling failed to start"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "failed to start profiler: {e} (a concurrent profiling \
                     request from someone else is the most likely cause — \
                     this profiler only runs one session at a time; retry \
                     shortly)"
                ),
            )
                .into_response();
        }
    };

    tokio::time::sleep(Duration::from_secs(params.seconds)).await;

    let report = match guard.report().build() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "cpu profiling failed to build report");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build profile report: {e}"),
            )
                .into_response();
        }
    };

    let mut svg = Vec::new();
    if let Err(e) = report.flamegraph(&mut svg) {
        tracing::warn!(error = %e, "cpu profiling failed to render flamegraph");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to render flamegraph: {e}"),
        )
            .into_response();
    }

    ([(header::CONTENT_TYPE, "image/svg+xml")], svg).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// Any `Deserializer` respects `#[serde(default = ...)]`, so this pins
    /// the default-value wiring without going through axum's query-string
    /// extractor specifically (and without invoking the profiler at all).
    #[test]
    fn defaults_apply_when_fields_absent() {
        let params: ProfileParams = serde_json::from_str("{}").unwrap();
        assert_eq!(params.seconds, 10);
        assert_eq!(params.frequency, 99);
    }

    #[test]
    fn fields_override_the_defaults() {
        let params: ProfileParams =
            serde_json::from_str(r#"{"seconds":5,"frequency":50}"#).unwrap();
        assert_eq!(params.seconds, 5);
        assert_eq!(params.frequency, 50);
    }

    /// End-to-end wiring check via the real router, at `seconds=1` (the
    /// minimum this route's own validation accepts) — exercises route
    /// registration, query extraction, and response construction with the
    /// smallest wait that clears that bound, same "fast, deterministic"
    /// reasoning as `health.rs`'s route tests, just for a route that can't be
    /// instant by design.
    #[tokio::test]
    async fn profile_route_returns_svg_flamegraph() {
        let app =
            crate::server::app::build_router(crate::server::app_context::AppContext::stub().into());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/debug/pprof/profile?seconds=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/svg+xml"
        );
    }

    #[tokio::test]
    async fn profile_route_rejects_out_of_range_frequency() {
        let app =
            crate::server::app::build_router(crate::server::app_context::AppContext::stub().into());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/debug/pprof/profile?seconds=1&frequency=0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
