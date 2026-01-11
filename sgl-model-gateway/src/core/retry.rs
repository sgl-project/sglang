use std::time::Duration;

use axum::http::StatusCode;
use rand::Rng;
use tracing::debug;

use crate::config::types::RetryConfig;

/// Check if an HTTP status code indicates a retryable error
pub fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

/// Computes exponential backoff with optional jitter.
#[derive(Debug, Clone)]
pub struct BackoffCalculator;

impl BackoffCalculator {
    /// Calculate backoff delay for a given attempt index (0-based).
    pub fn calculate_delay(config: &RetryConfig, attempt: u32) -> Duration {
        let pow = config.backoff_multiplier.powi(attempt as i32);
        let delay_ms = ((config.initial_backoff_ms as f32 * pow) as u64).min(config.max_backoff_ms);

        let jitter = config.jitter_factor.clamp(0.0, 1.0);
        if jitter > 0.0 {
            let mut rng = rand::rng();
            let jitter_scale: f32 = rng.random_range(-jitter..=jitter);
            let jitter_ms = (delay_ms as f32 * jitter_scale)
                .round()
                .max(-(delay_ms as f32));
            let adjusted = (delay_ms as i64 + jitter_ms as i64).max(0) as u64;
            return Duration::from_millis(adjusted);
        }

        Duration::from_millis(delay_ms)
    }
}

#[derive(Debug)]
pub struct MaxRetriesExceeded<T> {
    pub last: T,
}

/// A thin async retry executor for generic operations.
#[derive(Debug, Clone, Default)]
pub struct RetryExecutor;

impl RetryExecutor {
    /// Execute an async operation with retries and backoff.
    ///
    /// Returns `Ok(T)` on success, or `Err(MaxRetriesExceeded { last: T })` when exhausted.
    ///
    /// - `operation(attempt)`: perform one attempt (0-based), return the output.
    /// - `should_retry(&output, attempt)`: return true to retry, false to accept.
    /// - `on_backoff(&output, delay, next_attempt)`: called before sleeping between attempts.
    /// - `on_exhausted()`: called when retries are exhausted.
    pub async fn execute_with_retry<Op, Fut, T, ShouldRetry, OnBackoff, OnExhausted>(
        config: &RetryConfig,
        mut operation: Op,
        should_retry: ShouldRetry,
        on_backoff: OnBackoff,
        mut on_exhausted: OnExhausted,
    ) -> Result<T, MaxRetriesExceeded<T>>
    where
        Op: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = T>,
        ShouldRetry: Fn(&T, u32) -> bool,
        OnBackoff: Fn(&T, Duration, u32),
        OnExhausted: FnMut(),
    {
        let max = config.max_retries.max(1);
        let mut attempt: u32 = 0;

        loop {
            let output = operation(attempt).await;
            let is_last = attempt + 1 >= max;

            if !should_retry(&output, attempt) {
                return Ok(output);
            }

            if is_last {
                on_exhausted();
                return Err(MaxRetriesExceeded { last: output });
            }

            let next_attempt = attempt + 1;
            let delay = BackoffCalculator::calculate_delay(config, attempt);
            debug!(
                attempt = attempt,
                next_attempt = next_attempt,
                delay_ms = delay.as_millis() as u64,
                "Retry backoff"
            );
            on_backoff(&output, delay, next_attempt);
            tokio::time::sleep(delay).await;

            attempt = next_attempt;
        }
    }

    /// Like `execute_with_retry`, but returns the last output directly when exhausted.
    ///
    /// Useful for HTTP responses where you always need to return something.
    pub async fn execute_with_retry_or_last<Op, Fut, T, ShouldRetry, OnBackoff, OnExhausted>(
        config: &RetryConfig,
        operation: Op,
        should_retry: ShouldRetry,
        on_backoff: OnBackoff,
        on_exhausted: OnExhausted,
    ) -> T
    where
        Op: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = T>,
        ShouldRetry: Fn(&T, u32) -> bool,
        OnBackoff: Fn(&T, Duration, u32),
        OnExhausted: FnMut(),
    {
        Self::execute_with_retry(config, operation, should_retry, on_backoff, on_exhausted).await
            .unwrap_or_else(|MaxRetriesExceeded { last }| last)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    use axum::{http::StatusCode, response::IntoResponse};

    use super::*;

    fn base_retry_config() -> RetryConfig {
        RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 4,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
        }
    }

    #[test]
    fn test_backoff_no_jitter_progression_and_cap() {
        let cfg = RetryConfig {
            max_retries: 10,
            initial_backoff_ms: 100,
            max_backoff_ms: 250,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
        };
        assert_eq!(
            BackoffCalculator::calculate_delay(&cfg, 0),
            Duration::from_millis(100)
        );
        assert_eq!(
            BackoffCalculator::calculate_delay(&cfg, 1),
            Duration::from_millis(200)
        );
        assert_eq!(
            BackoffCalculator::calculate_delay(&cfg, 2),
            Duration::from_millis(250)
        );
        assert_eq!(
            BackoffCalculator::calculate_delay(&cfg, 10),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn test_backoff_with_jitter_within_bounds() {
        let cfg = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 100,
            max_backoff_ms: 10_000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.5,
        };
        let base = 400.0;
        for _ in 0..50 {
            let d = BackoffCalculator::calculate_delay(&cfg, 2).as_millis() as f32;
            assert!(d >= base * 0.5 - 1.0 && d <= base * 1.5 + 1.0);
        }
    }

    #[tokio::test]
    async fn test_execute_with_retry_success_after_failures() {
        let cfg = base_retry_config();
        let remaining = Arc::new(AtomicU32::new(2));
        let calls = Arc::new(AtomicU32::new(0));
        let backoffs = Arc::new(AtomicU32::new(0));

        let res = RetryExecutor::execute_with_retry(
            &cfg,
            {
                let remaining = remaining.clone();
                let calls = calls.clone();
                move |_attempt| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    let remaining = remaining.clone();
                    async move {
                        if remaining
                            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1))
                            .is_ok()
                        {
                            None
                        } else {
                            Some(42u32)
                        }
                    }
                }
            },
            |output, _attempt| output.is_none(),
            {
                let backoffs = backoffs.clone();
                move |_output, _delay, _next_attempt| {
                    backoffs.fetch_add(1, Ordering::Relaxed);
                }
            },
            || {},
        )
        .await;

        assert!(res.is_ok());
        assert_eq!(res.unwrap(), Some(42));
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(backoffs.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_execute_with_retry_exhausted() {
        let cfg = base_retry_config();
        let calls = Arc::new(AtomicU32::new(0));
        let exhausted = Arc::new(AtomicU32::new(0));

        let res = RetryExecutor::execute_with_retry(
            &cfg,
            {
                let calls = calls.clone();
                move |_attempt| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    async move { None::<u32> }
                }
            },
            |output, _attempt| output.is_none(),
            |_output, _delay, _next_attempt| {},
            {
                let exhausted = exhausted.clone();
                move || {
                    exhausted.fetch_add(1, Ordering::Relaxed);
                }
            },
        )
        .await;

        assert!(matches!(res, Err(MaxRetriesExceeded { last: None })));
        assert_eq!(calls.load(Ordering::Relaxed), cfg.max_retries);
        assert_eq!(exhausted.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_execute_with_retry_or_last_success_path_and_hooks() {
        let cfg = base_retry_config();
        let remaining = Arc::new(AtomicU32::new(2));
        let calls = Arc::new(AtomicU32::new(0));
        let backoffs = Arc::new(AtomicU32::new(0));
        let exhausted = Arc::new(AtomicU32::new(0));

        let response = RetryExecutor::execute_with_retry_or_last(
            &cfg,
            {
                let remaining = remaining.clone();
                let calls = calls.clone();
                move |_attempt| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    let remaining = remaining.clone();
                    async move {
                        if remaining
                            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1))
                            .is_ok()
                        {
                            (StatusCode::SERVICE_UNAVAILABLE, "fail").into_response()
                        } else {
                            (StatusCode::OK, "ok").into_response()
                        }
                    }
                }
            },
            |res, _attempt| !res.status().is_success(),
            {
                let backoffs = backoffs.clone();
                move |_output, _delay, _next_attempt| {
                    backoffs.fetch_add(1, Ordering::Relaxed);
                }
            },
            {
                let exhausted = exhausted.clone();
                move || {
                    exhausted.fetch_add(1, Ordering::Relaxed);
                }
            },
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(backoffs.load(Ordering::Relaxed), 2);
        assert_eq!(exhausted.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_execute_with_retry_or_last_non_retryable_short_circuit() {
        let cfg = base_retry_config();
        let calls = Arc::new(AtomicU32::new(0));
        let backoffs = Arc::new(AtomicU32::new(0));
        let exhausted = Arc::new(AtomicU32::new(0));

        let response = RetryExecutor::execute_with_retry_or_last(
            &cfg,
            {
                let calls = calls.clone();
                move |_attempt| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    async move { (StatusCode::BAD_REQUEST, "bad").into_response() }
                }
            },
            |_res, _attempt| false,
            {
                let backoffs = backoffs.clone();
                move |_output, _delay, _next_attempt| {
                    backoffs.fetch_add(1, Ordering::Relaxed);
                }
            },
            {
                let exhausted = exhausted.clone();
                move || {
                    exhausted.fetch_add(1, Ordering::Relaxed);
                }
            },
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(backoffs.load(Ordering::Relaxed), 0);
        assert_eq!(exhausted.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_execute_with_retry_or_last_exhausted_hooks() {
        let cfg = base_retry_config();
        let calls = Arc::new(AtomicU32::new(0));
        let backoffs = Arc::new(AtomicU32::new(0));
        let exhausted = Arc::new(AtomicU32::new(0));

        let response = RetryExecutor::execute_with_retry_or_last(
            &cfg,
            {
                let calls = calls.clone();
                move |_attempt| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    async move { (StatusCode::SERVICE_UNAVAILABLE, "fail").into_response() }
                }
            },
            |_res, _attempt| true,
            {
                let backoffs = backoffs.clone();
                move |_output, _delay, _next_attempt| {
                    backoffs.fetch_add(1, Ordering::Relaxed);
                }
            },
            {
                let exhausted = exhausted.clone();
                move || {
                    exhausted.fetch_add(1, Ordering::Relaxed);
                }
            },
        )
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(calls.load(Ordering::Relaxed), cfg.max_retries);
        assert_eq!(backoffs.load(Ordering::Relaxed), cfg.max_retries - 1);
        assert_eq!(exhausted.load(Ordering::Relaxed), 1);
    }
}
