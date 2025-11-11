use std::time::Duration;

use axum::{http::StatusCode, response::Response};
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
        let mut delay_ms = (config.initial_backoff_ms as f32 * pow) as u64;
        if delay_ms > config.max_backoff_ms {
            delay_ms = config.max_backoff_ms;
        }

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

#[derive(Debug, thiserror::Error)]
pub enum RetryError {
    #[error("no available workers")]
    NoAvailableWorkers,
    #[error("maximum retry attempts exceeded")]
    MaxRetriesExceeded,
}

/// A thin async retry executor for generic operations.
#[derive(Debug, Clone, Default)]
pub struct RetryExecutor;

impl RetryExecutor {
    /// Execute an async operation with retries and backoff.
    /// The `operation` closure is invoked each attempt with the attempt index.
    pub async fn execute_with_retry<F, Fut, T>(
        config: &RetryConfig,
        mut operation: F,
    ) -> Result<T, RetryError>
    where
        F: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = Result<T, ()>>,
    {
        let max = config.max_retries.max(1);
        let mut attempt: u32 = 0;
        loop {
            match operation(attempt).await {
                Ok(val) => return Ok(val),
                Err(_) => {
                    let is_last = attempt + 1 >= max;
                    if is_last {
                        return Err(RetryError::MaxRetriesExceeded);
                    }
                    let delay = BackoffCalculator::calculate_delay(config, attempt);
                    attempt += 1;
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Execute an operation that returns an HTTP Response with retries and backoff.
    ///
    /// Usage pattern:
    /// - `operation(attempt)`: perform one attempt (0-based). Construct and send the request,
    ///   then return the `Response`. Do any per-attempt bookkeeping (e.g., load tracking,
    ///   circuit-breaker outcome recording) inside this closure.
    /// - `should_retry(&response, attempt)`: decide if the given response should be retried
    ///   (e.g., based on HTTP status). Returning false short-circuits and returns the response.
    /// - `on_backoff(delay, next_attempt)`: called before sleeping between attempts.
    ///   Use this to record metrics.
    /// - `on_exhausted()`: called when the executor has exhausted all retry attempts.
    ///
    /// Example:
    /// ```ignore
    /// let resp = RetryExecutor::execute_response_with_retry(
    ///     &retry_cfg,
    ///     |attempt| async move {
    ///         let worker = select_cb_aware_worker()?;
    ///         let resp = send_request(worker).await;
    ///         worker.record_outcome(resp.status().is_success());
    ///         resp
    ///     },
    ///     |res, _| matches!(res.status(), StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS | StatusCode::INTERNAL_SERVER_ERROR | StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT),
    ///     |delay, attempt| RouterMetrics::record_retry_backoff_duration(delay, attempt),
    ///     || RouterMetrics::record_retries_exhausted("/route"),
    /// ).await;
    /// ```
    pub async fn execute_response_with_retry<Op, Fut, ShouldRetry, OnBackoff, OnExhausted>(
        config: &RetryConfig,
        mut operation: Op,
        should_retry: ShouldRetry,
        on_backoff: OnBackoff,
        mut on_exhausted: OnExhausted,
    ) -> Response
    where
        Op: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = Response>,
        ShouldRetry: Fn(&Response, u32) -> bool,
        OnBackoff: Fn(Duration, u32),
        OnExhausted: FnMut(),
    {
        let max = config.max_retries.max(1);

        let mut attempt: u32 = 0;
        loop {
            let response = operation(attempt).await;
            let is_last = attempt + 1 >= max;

            if !should_retry(&response, attempt) {
                return response;
            }

            if is_last {
                on_exhausted();
                return response;
            }

            let next_attempt = attempt + 1;
            let delay = BackoffCalculator::calculate_delay(config, attempt);
            debug!(
                attempt = attempt,
                next_attempt = next_attempt,
                delay_ms = delay.as_millis() as u64,
                "Retry backoff"
            );
            on_backoff(delay, next_attempt);
            tokio::time::sleep(delay).await;

            attempt = next_attempt;
        }
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

        let res: Result<u32, RetryError> = RetryExecutor::execute_with_retry(&cfg, {
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
                        Err(())
                    } else {
                        Ok(42u32)
                    }
                }
            }
        })
        .await;

        assert!(res.is_ok());
        assert_eq!(res.unwrap(), 42);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_execute_with_retry_exhausted() {
        let cfg = base_retry_config();
        let calls = Arc::new(AtomicU32::new(0));
        let res: Result<u32, RetryError> = RetryExecutor::execute_with_retry(&cfg, {
            let calls = calls.clone();
            move |_attempt| {
                calls.fetch_add(1, Ordering::Relaxed);
                async move { Err(()) }
            }
        })
        .await;

        assert!(matches!(res, Err(RetryError::MaxRetriesExceeded)));
        assert_eq!(calls.load(Ordering::Relaxed), cfg.max_retries);
    }

    #[tokio::test]
    async fn test_execute_response_with_retry_success_path_and_hooks() {
        let cfg = base_retry_config();
        let remaining = Arc::new(AtomicU32::new(2));
        let calls = Arc::new(AtomicU32::new(0));
        let backoffs = Arc::new(AtomicU32::new(0));
        let exhausted = Arc::new(AtomicU32::new(0));

        let response = RetryExecutor::execute_response_with_retry(
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
                move |_delay, _next_attempt| {
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
    async fn test_execute_response_with_retry_non_retryable_short_circuit() {
        let cfg = base_retry_config();
        let calls = Arc::new(AtomicU32::new(0));
        let backoffs = Arc::new(AtomicU32::new(0));
        let exhausted = Arc::new(AtomicU32::new(0));

        let response = RetryExecutor::execute_response_with_retry(
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
                move |_delay, _next_attempt| {
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
    async fn test_execute_response_with_retry_exhausted_hooks() {
        let cfg = base_retry_config();
        let calls = Arc::new(AtomicU32::new(0));
        let backoffs = Arc::new(AtomicU32::new(0));
        let exhausted = Arc::new(AtomicU32::new(0));

        let response = RetryExecutor::execute_response_with_retry(
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
                move |_delay, _next_attempt| {
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
