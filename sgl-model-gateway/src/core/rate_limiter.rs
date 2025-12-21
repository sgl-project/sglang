use std::sync::Arc;

use dashmap::DashMap;
use tracing::{debug, info};

use crate::{config::RouterConfig, middleware::TokenBucket};

/// Key for rate limit buckets: (tenant_id, model_id)
type RateLimitKey = (Option<String>, Option<String>);

/// Multi-tenant and model-specific rate limiter
pub struct RateLimiter {
    /// Map of buckets keyed by (tenant_id, model_id)
    buckets: DashMap<RateLimitKey, Arc<TokenBucket>>,
    /// Default bucket settings if no rule matches
    default_max_concurrent: i32,
    default_refill_rate: Option<i32>,
}

impl RateLimiter {
    /// Create a new rate limiter from configuration
    pub fn new(config: &RouterConfig) -> Self {
        let limiter = Self {
            buckets: DashMap::new(),
            default_max_concurrent: config.max_concurrent_requests,
            default_refill_rate: config.rate_limit_tokens_per_second,
        };

        // Initialize buckets from rules
        if let Some(rules) = &config.rate_limits {
            for rule in rules {
                let capacity = rule.max_concurrent_requests;
                let refill_rate = rule
                    .rate_limit_tokens_per_second
                    .filter(|&r| r > 0)
                    .unwrap_or(capacity);

                let bucket = Arc::new(TokenBucket::new(capacity as usize, refill_rate as usize));
                let key = (rule.tenant_id.clone(), rule.model_id.clone());

                info!(
                    "Initialized rate limit bucket for tenant={:?}, model={:?} (capacity={}, refill={})",
                    rule.tenant_id, rule.model_id, capacity, refill_rate
                );
                limiter.buckets.insert(key, bucket);
            }
        }

        limiter
    }

    /// Get the appropriate token bucket for a request based on tenant and model
    pub fn get_bucket(
        &self,
        tenant_id: Option<&str>,
        model_id: Option<&str>,
    ) -> Option<Arc<TokenBucket>> {
        let tenant = tenant_id.map(|s| s.to_string());
        let model = model_id.map(|s| s.to_string());

        // 1. Try specific match (tenant + model)
        if let Some(bucket) = self.buckets.get(&(tenant.clone(), model.clone())) {
            return Some(Arc::clone(bucket.value()));
        }

        // 2. Try tenant-only match
        if tenant.is_some() {
            if let Some(bucket) = self.buckets.get(&(tenant.clone(), None)) {
                return Some(Arc::clone(bucket.value()));
            }
        }

        // 3. Try model-only match
        if model.is_some() {
            if let Some(bucket) = self.buckets.get(&(None, model.clone())) {
                return Some(Arc::clone(bucket.value()));
            }
        }

        // 4. Fallback to global default if enabled
        if self.default_max_concurrent > 0 {
            // Check if we already have a global bucket
            if let Some(bucket) = self.buckets.get(&(None, None)) {
                return Some(Arc::clone(bucket.value()));
            }

            // Create global bucket if it doesn't exist
            let refill_rate = self
                .default_refill_rate
                .filter(|&r| r > 0)
                .unwrap_or(self.default_max_concurrent);

            let bucket = Arc::new(TokenBucket::new(
                self.default_max_concurrent as usize,
                refill_rate as usize,
            ));

            debug!(
                "Created global default rate limit bucket (capacity={}, refill={})",
                self.default_max_concurrent, refill_rate
            );

            self.buckets.insert((None, None), Arc::clone(&bucket));
            return Some(bucket);
        }

        None
    }

    /// Check if rate limiting is enabled (either via rules or global default)
    pub fn is_enabled(&self) -> bool {
        self.default_max_concurrent > 0 || !self.buckets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RateLimitRule;

    #[tokio::test]
    async fn test_rate_limiter_hierarchy() {
        let rules = vec![
            // Specific rule
            RateLimitRule {
                tenant_id: Some("tenant-a".to_string()),
                model_id: Some("model-1".to_string()),
                max_concurrent_requests: 1,
                rate_limit_tokens_per_second: None,
            },
            // Tenant-only rule
            RateLimitRule {
                tenant_id: Some("tenant-a".to_string()),
                model_id: None,
                max_concurrent_requests: 5,
                rate_limit_tokens_per_second: None,
            },
            // Model-only rule
            RateLimitRule {
                tenant_id: None,
                model_id: Some("model-2".to_string()),
                max_concurrent_requests: 10,
                rate_limit_tokens_per_second: None,
            },
        ];

        let config = RouterConfig {
            max_concurrent_requests: 100,
            rate_limits: Some(rules),
            ..Default::default()
        };

        let limiter = RateLimiter::new(&config);

        // 1. Specific match: tenant-a + model-1 -> bucket with 1 token
        let b = limiter
            .get_bucket(Some("tenant-a"), Some("model-1"))
            .unwrap();
        assert_eq!(b.available_tokens().await, 1.0);

        // 2. Tenant fallback: tenant-a + unknown model -> bucket with 5 tokens
        let b = limiter
            .get_bucket(Some("tenant-a"), Some("unknown"))
            .unwrap();
        assert_eq!(b.available_tokens().await, 5.0);

        // 3. Model fallback: unknown tenant + model-2 -> bucket with 10 tokens
        let b = limiter.get_bucket(None, Some("model-2")).unwrap();
        assert_eq!(b.available_tokens().await, 10.0);

        // 4. Global fallback: unknown tenant + unknown model -> bucket with 100 tokens
        let b = limiter.get_bucket(None, None).unwrap();
        assert_eq!(b.available_tokens().await, 100.0);
    }

    #[test]
    fn test_rate_limiter_disabled() {
        let config = RouterConfig {
            max_concurrent_requests: -1,
            rate_limits: None,
            ..Default::default()
        };
        let limiter = RateLimiter::new(&config);
        assert!(!limiter.is_enabled());
        assert!(limiter.get_bucket(None, None).is_none());
    }
}
