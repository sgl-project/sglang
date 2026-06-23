//! Tests for the real in-flight request cap (`max_inflight_requests`):
//! a true semaphore (refill_rate=0), distinct from `max_concurrent_requests`
//! (a req/s bucket). Default 0 = disabled.

mod common;

use std::sync::Arc;

use smg::config::{builder::RouterConfigBuilder, RouterConfig};

/// A builder with sane defaults, old req/s limiter disabled so it can't
/// mask the new in-flight cap. Callers add `.max_inflight_requests(...)`
/// then `.build()` / `.build_unchecked()` as needed.
fn base_config() -> RouterConfigBuilder {
    RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(30100)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        // Keep the old req/s limiter disabled so it can't mask the new cap.
        .disable_rate_limiting()
        .queue_timeout_secs(60)
}

#[tokio::test]
async fn inflight_cap_field_defaults_to_zero_and_is_settable() {
    let cfg = base_config().build_unchecked();
    assert_eq!(
        cfg.max_inflight_requests, 0,
        "default must be 0 (disabled) to preserve existing behavior"
    );

    let cfg = base_config().max_inflight_requests(8).build_unchecked();
    assert_eq!(cfg.max_inflight_requests, 8);

    let cfg = base_config().max_inflight_requests(-1).build_unchecked();
    assert_eq!(
        cfg.max_inflight_requests, -1,
        "-1 must be accepted (explicit disable)"
    );
}

#[tokio::test]
async fn inflight_cap_field_serde_roundtrips() {
    let cfg = base_config().max_inflight_requests(16).build_unchecked();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: RouterConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.max_inflight_requests, 16);

    // Omitting the field must deserialize to 0 (serde default).
    let cfg = base_config().build_unchecked();
    let minimal = serde_json::to_value(&cfg).expect("to_value");
    let stripped = {
        let mut v = minimal.clone();
        if let serde_json::Value::Object(ref mut m) = v {
            m.remove("max_inflight_requests");
        }
        v
    };
    let from_stripped: RouterConfig = serde_json::from_value(stripped).expect("deserialize");
    assert_eq!(
        from_stripped.max_inflight_requests, 0,
        "missing field must default to 0, not fail"
    );
}

#[tokio::test]
async fn inflight_cap_validation_rejects_below_minus_one() {
    // -1 and 0 are valid; -2 must be rejected by `build()` (validated path).
    let _ok_neg1 = base_config()
        .max_inflight_requests(-1)
        .build()
        .expect("-1 is valid (explicit disable)");
    let _ok_zero = base_config().build().expect("0 is valid (default)");

    let err = base_config()
        .max_inflight_requests(-2)
        .build()
        .expect_err("-2 must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("max_inflight_requests"),
        "error should name the field: {msg}"
    );
}

#[tokio::test]
async fn inflight_limiter_is_a_true_semaphore_when_enabled() {
    let cfg = base_config().max_inflight_requests(2).build_unchecked();
    let ctx = common::create_test_context(cfg).await;

    let limiter = ctx
        .inflight_limiter
        .clone()
        .expect("inflight_limiter must be Some when max_inflight_requests > 0");

    assert!(limiter.try_acquire(1.0).await.is_ok(), "first acquire");
    assert!(limiter.try_acquire(1.0).await.is_ok(), "second acquire");

    // No refill: a semaphore, not a req/s bucket.
    assert!(
        limiter.try_acquire(1.0).await.is_err(),
        "third try_acquire must fail while two are held (refill_rate must be 0)"
    );

    limiter.return_tokens_sync(1.0);
    assert!(
        limiter.try_acquire(1.0).await.is_ok(),
        "acquire must succeed after a permit is returned"
    );
}

#[tokio::test]
async fn inflight_limiter_is_none_when_disabled() {
    for value in [0, -1] {
        let cfg = base_config().max_inflight_requests(value).build_unchecked();
        let ctx = common::create_test_context(cfg).await;
        assert!(
            ctx.inflight_limiter.is_none(),
            "inflight_limiter must be None for max_inflight_requests = {value}"
        );
    }
}

// ---------------------------------------------------------------------------
// Regression: max_concurrent_requests still behaves as a req/s limiter and
// is NOT repurposed into an in-flight cap. With the default refill (tokens_per
// second == n) the bucket refills on a timer, so an (N+1)th acquire that would
// fail for a pure semaphore eventually succeeds WITHOUT any request finishing.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn max_concurrent_requests_remains_a_rate_limiter_not_a_semaphore() {
    use smg::middleware::TokenBucket;

    // Reproduce the prod wiring: TokenBucket::new(n, n) refills on a timer.
    let n = 4usize;
    let bucket = Arc::new(TokenBucket::new(n, n));

    for _ in 0..n {
        assert!(bucket.try_acquire(1.0).await.is_ok(), "burst acquire");
    }
    assert!(
        bucket.try_acquire(1.0).await.is_err(),
        "bucket must be empty right after exhausting burst"
    );

    // Wait WITHOUT returning a token. A req/s limiter refills on its own;
    // a semaphore (refill_rate=0) would never recover here.
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert!(
        bucket.try_acquire(1.0).await.is_ok(),
        "max_concurrent_requests must refill on a timer (req/s), proving it is \
         NOT a true in-flight cap"
    );
}

#[tokio::test]
async fn inflight_semaphore_does_not_refill_on_a_timer() {
    use smg::middleware::TokenBucket;

    // Reproduce the prod wiring: TokenBucket::new(n, 0) never refills.
    let bucket = Arc::new(TokenBucket::new(2, 0));

    assert!(bucket.try_acquire(1.0).await.is_ok());
    assert!(bucket.try_acquire(1.0).await.is_ok());
    assert!(bucket.try_acquire(1.0).await.is_err(), "capacity exhausted");

    // A req/s limiter (refill > 0) would have refilled by now.
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert!(
        bucket.try_acquire(1.0).await.is_err(),
        "in-flight semaphore must NOT refill on a timer; only return_tokens frees a slot"
    );

    bucket.return_tokens_sync(1.0);
    assert!(bucket.try_acquire(1.0).await.is_ok());
}
