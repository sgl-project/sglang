// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::time::Duration;

fn cb() -> CircuitBreaker {
    CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(3).unwrap(),
        cool_down: Duration::from_millis(100),
    })
}

#[test]
fn starts_closed_and_allows() {
    let b = cb();
    assert!(b.allow());
}

#[test]
fn three_failures_open_the_breaker() {
    let b = cb();
    b.record_failure();
    b.record_failure();
    assert!(b.allow(), "still closed before threshold");
    b.record_failure();
    assert!(!b.allow(), "open after threshold reached");
}

#[test]
fn intermittent_success_resets_failure_count() {
    let b = cb();
    b.record_failure();
    b.record_failure();
    b.record_success(); // resets
    b.record_failure();
    b.record_failure();
    assert!(b.allow(), "should still be closed (2 failures since reset)");
}

#[tokio::test(start_paused = true)]
async fn open_breaker_recovers_via_half_open() {
    let b = cb();
    b.record_failure();
    b.record_failure();
    b.record_failure();
    assert!(!b.allow());

    // Wait past cool_down.
    tokio::time::advance(Duration::from_millis(150)).await;

    // Half-open: allow one probe.
    assert!(b.allow(), "half-open allows the probe");
    // While half-open, further allow() calls should reject (only one probe in flight).
    assert!(!b.allow(), "half-open rejects second probe");

    // Probe succeeded.
    b.record_success();
    assert!(b.allow(), "closed after successful probe");
    assert!(b.allow(), "stays closed");
}

#[tokio::test(start_paused = true)]
async fn half_open_failure_reopens() {
    let b = cb();
    b.record_failure();
    b.record_failure();
    b.record_failure();

    tokio::time::advance(Duration::from_millis(150)).await;
    assert!(b.allow(), "half-open admit");
    b.record_failure();
    // Back to Open.
    assert!(!b.allow(), "back to open");
}

#[tokio::test(start_paused = true)]
async fn would_allow_is_non_mutating_past_cool_down() {
    // `would_allow()` answers "would `allow()` return true right now?" without
    // claiming a probe slot. Enumeration / filtering paths (e.g.
    // `WorkerRegistry::healthy_workers_for`) call it to inspect breakers
    // without disturbing state.
    let b = cb();
    b.record_failure();
    b.record_failure();
    b.record_failure();
    assert!(!b.allow(), "open after threshold");

    tokio::time::advance(Duration::from_millis(150)).await;

    // Repeated would_allow() returns true and leaves state untouched.
    assert!(b.would_allow());
    assert!(b.would_allow());
    assert!(b.would_allow());

    // The first allow() claims the half-open probe.
    assert!(b.allow(), "allow() admits the probe");
    // The probe is in flight — subsequent allow() (and would_allow()) reject.
    assert!(!b.allow(), "only one probe in flight");
    assert!(!b.would_allow(), "would_allow() agrees: no slot available");
}

#[tokio::test(start_paused = true)]
async fn enumeration_then_dispatch_preserves_probe() {
    // Regression for the bug where `healthy_workers_for` filtered with
    // mutating `allow()`. Once would_allow() is the filter, an enumeration
    // pass over many workers must not steal the probe slot from the one
    // worker that actually gets dispatched to.
    let b = cb();
    b.record_failure();
    b.record_failure();
    b.record_failure();
    tokio::time::advance(Duration::from_millis(150)).await;

    // Imagine 3 workers; enumeration filters each with would_allow().
    for _ in 0..3 {
        assert!(b.would_allow(), "filter sees the worker as available");
    }

    // Now the policy picks ONE worker and dispatch claims the probe.
    assert!(b.allow(), "dispatch on the picked worker succeeds");
}

#[test]
fn would_allow_in_closed_state_is_true_and_non_mutating() {
    let b = cb();
    for _ in 0..5 {
        assert!(b.would_allow());
    }
    // And allow() should still work afterwards.
    assert!(b.allow());
}

#[tokio::test(start_paused = true)]
async fn open_breaker_recovery_is_not_delayed_by_continued_failures() {
    // Regression: previously, record_failure on an already-Open breaker
    // refreshed opened_at, so a failure storm pinned the breaker open
    // forever. Now the cool_down is measured from first-open.
    let b = CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(3).unwrap(),
        cool_down: Duration::from_millis(100),
    });
    // Open it.
    b.record_failure();
    b.record_failure();
    b.record_failure();
    assert!(!b.allow(), "breaker should be open");

    // Advance halfway through cool_down, then record more failures.
    tokio::time::advance(Duration::from_millis(50)).await;
    b.record_failure();
    b.record_failure();
    b.record_failure();

    // Advance just past the original cool_down.
    tokio::time::advance(Duration::from_millis(60)).await;

    // We're past the original cool_down → HalfOpen.
    assert!(
        b.allow(),
        "breaker should be half-open after cool_down from first-open"
    );
}
