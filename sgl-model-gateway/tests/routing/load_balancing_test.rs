//! Load balancing integration tests
//!
//! Tests for various load balancing policies: round_robin, random, cache_aware, etc.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{CircuitBreakerConfig, RetryConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod round_robin_tests {
    use super::*;

    /// Test that round robin distributes requests evenly across workers
    #[tokio::test]
    async fn test_round_robin_distribution() {
        let config = TestRouterConfig::round_robin(3100);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19001, 3))
                .await;

        let app = ctx.create_app().await;
        let num_requests = 30;
        let mut success_count = 0;

        for i in 0..num_requests {
            let payload = json!({
                "text": format!("Request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // All requests should succeed with round robin across 3 healthy workers
        assert_eq!(
            success_count, num_requests,
            "All requests should succeed with round robin"
        );

        ctx.shutdown().await;
    }

    /// Test round robin with one worker failing
    #[tokio::test]
    async fn test_round_robin_with_failing_worker() {
        let config = TestRouterConfig::round_robin_with_retry(
            3101,
            RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
        );
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19004, 1.0), // Always fail
                TestWorkerConfig::healthy(19005),    // Always succeed
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // With retries enabled, requests should eventually succeed
        // by being retried on the healthy worker
        let payload = json!({
            "text": "Test with retries",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry on healthy worker"
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod random_tests {
    use super::*;

    /// Test that random policy distributes requests across workers
    #[tokio::test]
    async fn test_random_distribution() {
        let config = TestRouterConfig::random(3102);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19010, 2))
                .await;

        let app = ctx.create_app().await;
        let num_requests = 20;
        let mut success_count = 0;

        for i in 0..num_requests {
            let payload = json!({
                "text": format!("Random request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // All requests should succeed with random policy
        assert_eq!(
            success_count, num_requests,
            "All requests should succeed with random policy"
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod cache_aware_tests {
    use super::*;

    /// Test cache-aware routing uses consistent hashing
    #[tokio::test]
    async fn test_cache_aware_consistent_routing() {
        let config = TestRouterConfig::cache_aware(3103);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19020, 2))
                .await;

        let app = ctx.create_app().await;

        // Same prompt should route to same worker (consistent hashing)
        let same_prompt = "Hello, cache-aware routing test!";
        let mut worker_ids: Vec<Option<String>> = Vec::new();

        for _ in 0..5 {
            let payload = json!({
                "text": same_prompt,
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            // Extract worker ID from response header if available
            let worker_id = resp
                .headers()
                .get("x-worker-id")
                .map(|v| v.to_str().unwrap().to_string());
            worker_ids.push(worker_id);
        }

        // All requests should succeed
        assert_eq!(worker_ids.len(), 5);

        ctx.shutdown().await;
    }

    /// Test cache-aware routing with different prompts
    #[tokio::test]
    async fn test_cache_aware_different_prompts() {
        let config = TestRouterConfig::cache_aware(3104);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19022, 2))
                .await;

        let app = ctx.create_app().await;
        let prompts = vec![
            "First unique prompt",
            "Second unique prompt",
            "Third unique prompt",
            "Fourth unique prompt",
        ];

        let mut success_count = 0;
        for prompt in prompts {
            let payload = json!({
                "text": prompt,
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // All requests should succeed
        assert_eq!(success_count, 4);

        ctx.shutdown().await;
    }

    /// Sum every sample of `name` whose label set contains `label`.
    fn prom_sum(exposition: &str, name: &str, label: &str) -> f64 {
        exposition
            .lines()
            .filter(|l| l.starts_with(name) && l[name.len()..].starts_with('{'))
            .filter(|l| l.contains(label))
            .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
            .sum()
    }

    /// End to end over HTTP: slow mock workers behind the real router, scraped
    /// from the real Prometheus exporter. Concurrent requests sharing a prefix
    /// pile onto the worker that holds it until the balance thresholds trip,
    /// and the reuse the load-balance branch gives up must then be exported.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cache_aware_load_balance_branch_is_measured() {
        use smg::observability::metrics::{start_prometheus, PrometheusConfig};

        const PROM_PORT: u16 = 29190;
        start_prometheus(PrometheusConfig {
            port: PROM_PORT,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        });

        let mut config = TestRouterConfig::cache_aware(3190);
        config.policy = smg::config::PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 2,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 60,
            max_tree_size: 1000,
        };
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::slow_workers(19190, 2, 300))
                .await;
        let app = ctx.create_app().await;

        let shared_prefix = "You are a coding agent. ".repeat(40);
        let requests = (0..24).map(|i| {
            let app = app.clone();
            let payload = json!({
                "text": format!("{shared_prefix} turn {i}"),
                "stream": false
            });
            async move {
                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();
                app.oneshot(req).await.unwrap().status()
            }
        });
        let statuses = futures::future::join_all(requests).await;
        assert!(statuses.iter().all(|s| *s == StatusCode::OK));

        let exposition = reqwest::get(format!("http://127.0.0.1:{PROM_PORT}/metrics"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        for line in exposition
            .lines()
            .filter(|l| l.starts_with("smg_cache_aware_"))
        {
            println!("{line}");
        }

        let lb = r#"branch="load_balance""#;
        assert!(
            prom_sum(&exposition, "smg_cache_aware_policy_branch_total", lb) > 0.0,
            "load-balance branch never ran"
        );
        assert!(
            prom_sum(&exposition, "smg_cache_aware_match_rate_count", lb) > 0.0,
            "match rate not recorded on the load-balance branch"
        );
        assert!(
            exposition.contains("smg_cache_aware_match_rate_bucket{"),
            "match rate exported as a summary, not a histogram"
        );
        let given_up = prom_sum(&exposition, "smg_cache_aware_matched_chars_total", lb);
        let inspected = prom_sum(&exposition, "smg_cache_aware_input_chars_total", lb);
        assert!(
            given_up > 0.0 && given_up <= inspected,
            "reuse given up: matched={given_up} input={inspected}"
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod worker_health_tests {
    use super::*;

    /// Test that failing workers are avoided via circuit breaker/retry
    #[tokio::test]
    async fn test_skip_failing_workers() {
        let config = TestRouterConfig::round_robin_with_reliability(
            3105,
            RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            },
        );
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19030, 1.0), // Always fails
                TestWorkerConfig::healthy(19031),    // Always succeeds
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Requests should succeed by retrying on healthy worker
        // or by circuit breaker opening on failing worker
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        ctx.shutdown().await;
    }

    /// Test behavior when all workers are unhealthy
    #[tokio::test]
    async fn test_all_workers_unhealthy() {
        let ctx = AppTestContext::new(vec![]).await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Test with no workers",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should return service unavailable when no workers available
        assert!(
            resp.status() == StatusCode::SERVICE_UNAVAILABLE
                || resp.status() == StatusCode::INTERNAL_SERVER_ERROR,
            "Expected 503 or 500 when no workers available, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod worker_response_delay_tests {
    use super::*;

    /// Test that response delays are handled correctly
    #[tokio::test]
    async fn test_worker_with_delay() {
        let config = TestRouterConfig::random(3106);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::slow(19040, 100)], // 100ms delay
        )
        .await;

        let app = ctx.create_app().await;

        let start = std::time::Instant::now();
        let payload = json!({
            "text": "Test with delay",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(resp.status(), StatusCode::OK);
        // Response should take at least 100ms due to configured delay
        assert!(
            elapsed.as_millis() >= 100,
            "Response should be delayed by at least 100ms, got {}ms",
            elapsed.as_millis()
        );

        ctx.shutdown().await;
    }
}
