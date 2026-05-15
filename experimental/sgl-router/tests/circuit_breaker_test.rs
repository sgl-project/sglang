// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::time::Duration;

fn cb() -> CircuitBreaker {
    CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: 3,
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
