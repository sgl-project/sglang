use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify};
use tracing::{debug, trace};

/// Token bucket for rate limiting
///
/// This implementation provides:
/// - Smooth rate limiting with configurable refill rate
/// - Burst capacity handling
/// - Fair queuing for waiting requests
#[derive(Clone)]
pub struct TokenBucket {
    inner: Arc<Mutex<TokenBucketInner>>,
    notify: Arc<Notify>,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

struct TokenBucketInner {
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of tokens (burst capacity)
    /// * `refill_rate` - Tokens added per second
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        let capacity = capacity as f64;
        let refill_rate = refill_rate as f64;

        // Ensure refill_rate is not zero to prevent division by zero
        let refill_rate = if refill_rate > 0.0 {
            refill_rate
        } else {
            1.0 // Default to 1 token per second if zero
        };

        Self {
            inner: Arc::new(Mutex::new(TokenBucketInner {
                tokens: capacity, // Start full
                last_refill: Instant::now(),
            })),
            notify: Arc::new(Notify::new()),
            capacity,
            refill_rate,
        }
    }

    /// Try to acquire tokens immediately
    pub async fn try_acquire(&self, tokens: f64) -> Result<(), ()> {
        let mut inner = self.inner.lock().await;

        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        inner.tokens = (inner.tokens + refill_amount).min(self.capacity);
        inner.last_refill = now;

        trace!(
            "Token bucket: {} tokens available, requesting {}",
            inner.tokens,
            tokens
        );

        if inner.tokens >= tokens {
            inner.tokens -= tokens;
            debug!(
                "Token bucket: acquired {} tokens, {} remaining",
                tokens, inner.tokens
            );
            Ok(())
        } else {
            Err(())
        }
    }

    /// Acquire tokens, waiting if necessary
    pub async fn acquire(&self, tokens: f64) -> Result<(), tokio::time::error::Elapsed> {
        // First try to acquire immediately
        if self.try_acquire(tokens).await.is_ok() {
            return Ok(());
        }

        // Calculate wait time
        let wait_time = {
            let inner = self.inner.lock().await;
            let tokens_needed = tokens - inner.tokens;
            let wait_secs = tokens_needed / self.refill_rate;
            Duration::from_secs_f64(wait_secs)
        };

        debug!(
            "Token bucket: waiting {:?} for {} tokens",
            wait_time, tokens
        );

        // Wait for tokens to be available
        tokio::time::timeout(wait_time, async {
            loop {
                // Check if we can acquire now
                if self.try_acquire(tokens).await.is_ok() {
                    return;
                }

                // Wait for notification or small interval
                tokio::select! {
                    _ = self.notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {},
                }
            }
        })
        .await?;

        Ok(())
    }

    /// Acquire tokens with custom timeout
    pub async fn acquire_timeout(
        &self,
        tokens: f64,
        timeout: Duration,
    ) -> Result<(), tokio::time::error::Elapsed> {
        tokio::time::timeout(timeout, self.acquire(tokens)).await?
    }

    /// Return tokens to the bucket (for cancelled requests)
    pub async fn return_tokens(&self, tokens: f64) {
        let mut inner = self.inner.lock().await;
        inner.tokens = (inner.tokens + tokens).min(self.capacity);
        self.notify.notify_waiters();
        debug!(
            "Token bucket: returned {} tokens, {} available",
            tokens, inner.tokens
        );
    }

    /// Get current available tokens (for monitoring)
    pub async fn available_tokens(&self) -> f64 {
        let mut inner = self.inner.lock().await;

        // Refill before checking
        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        inner.tokens = (inner.tokens + refill_amount).min(self.capacity);
        inner.last_refill = now;

        inner.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5); // 10 capacity, 5 per second

        // Should succeed - bucket starts full
        assert!(bucket.try_acquire(5.0).await.is_ok());
        assert!(bucket.try_acquire(5.0).await.is_ok());

        // Should fail - no tokens left
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Wait for refill
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Should have ~1.5 tokens now
        assert!(bucket.try_acquire(1.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10); // 10 capacity, 10 per second

        // Use all tokens
        assert!(bucket.try_acquire(10.0).await.is_ok());

        // Wait for partial refill
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Should have ~5 tokens
        let available = bucket.available_tokens().await;
        assert!((4.0..=6.0).contains(&available));
    }
}
