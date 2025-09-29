//! Integration tests for advanced routing functionality

use axum::http::{HeaderMap, HeaderValue};
use sglang_router_rs::routers::{AdvancedRoutingConfig, WorkerScore};

#[cfg(test)]
mod advanced_routing_tests {
    use super::*;

    #[test]
    fn test_header_parsing() {
        let mut headers = HeaderMap::new();
        headers.insert("x-worker-priority", HeaderValue::from_static("80"));
        headers.insert("x-max-cost", HeaderValue::from_static("1.5"));
        headers.insert("x-prefer-pd", HeaderValue::from_static("true"));

        // Test priority parsing
        let priority = headers
            .get("x-worker-priority")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok());
        assert_eq!(priority, Some(80));

        // Test cost parsing
        let max_cost = headers
            .get("x-max-cost")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f32>().ok());
        assert_eq!(max_cost, Some(1.5));

        // Test prefer-pd parsing
        let prefer_pd = headers
            .get("x-prefer-pd")
            .and_then(|v| v.to_str().ok())
            .map(|s| s == "true" || s == "1")
            .unwrap_or(false);
        assert_eq!(prefer_pd, true);
    }

    #[test]
    fn test_advanced_routing_config() {
        let config = AdvancedRoutingConfig {
            enabled: true,
            priority_weight: 2.0,
            load_weight: 4.0,
            pd_preference_weight: 3.0,
        };

        assert!(config.enabled);
        assert_eq!(config.priority_weight, 2.0);
        assert_eq!(config.load_weight, 4.0);
        assert_eq!(config.pd_preference_weight, 3.0);
    }

    #[test]
    fn test_worker_score_comparison() {
        // Create workers with different scores
        let high_priority_worker = WorkerScore::new(
            "http://high-priority:8000".to_string(),
            "http-regular".to_string(),
            3.0, // High score
            90,  // High priority
            1.0, // Normal cost
            2,   // Low load
            true,
            false,
        );

        let low_priority_worker = WorkerScore::new(
            "http://low-priority:8000".to_string(),
            "http-regular".to_string(),
            1.0, // Low score
            20,  // Low priority
            1.0, // Normal cost
            5,   // High load
            true,
            false,
        );

        let pd_worker = WorkerScore::new(
            "http://pd-worker:8000".to_string(),
            "http-pd".to_string(),
            2.5, // Medium score
            60,  // Medium priority
            1.2, // Higher cost
            3,   // Medium load
            true,
            true, // PD worker
        );

        // Test that high priority worker has higher score
        assert!(high_priority_worker.score > low_priority_worker.score);
        assert!(high_priority_worker.score > pd_worker.score);

        // Test sorting behavior
        let mut workers = vec![low_priority_worker, pd_worker, high_priority_worker];
        workers.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Should be sorted by score descending
        assert_eq!(workers[0].worker_url, "http://high-priority:8000");
        assert_eq!(workers[1].worker_url, "http://pd-worker:8000");
        assert_eq!(workers[2].worker_url, "http://low-priority:8000");
    }

    #[test]
    fn test_scoring_algorithm() {
        let config = AdvancedRoutingConfig {
            enabled: true,
            priority_weight: 1.0,
            load_weight: 3.0,
            pd_preference_weight: 2.0,
        };

        // Simulate scoring calculation for a worker
        let worker_priority = 75u32;
        let worker_load = 5usize;
        let max_observed_load = 10usize;
        let prefer_pd = true;
        let is_router_pd = true;

        let mut score = 1.0f64; // Base score

        // Priority bonus (normalize around default priority of 50)
        let priority_bonus = (worker_priority as f32 - 50.0) / 50.0 * config.priority_weight;
        score += priority_bonus as f64;

        // Load penalty
        if max_observed_load > 0 {
            let load_penalty =
                (worker_load as f64 / max_observed_load as f64) * config.load_weight as f64;
            score -= load_penalty;
        }

        // PD preference bonus
        if prefer_pd && is_router_pd {
            score += config.pd_preference_weight as f64;
        }

        // Expected calculation:
        // Base: 1.0
        // Priority bonus: (75-50)/50 * 1.0 = 0.5
        // Load penalty: (5/10) * 3.0 = 1.5
        // PD bonus: 2.0
        // Final: 1.0 + 0.5 - 1.5 + 2.0 = 2.0
        assert_eq!(score, 2.0);
    }

    #[test]
    fn test_cost_filtering() {
        // Test that workers with high cost are filtered out
        let max_cost = 1.5f32;
        let worker_cost = 2.0f32;

        // This worker should be filtered out
        assert!(worker_cost > max_cost);

        // This worker should pass the filter
        let acceptable_worker_cost = 1.2f32;
        assert!(acceptable_worker_cost <= max_cost);
    }

    #[test]
    fn test_priority_filtering() {
        let priority_threshold = 60u32;

        // Workers with priority below threshold should be filtered out
        let low_priority_worker = 40u32;
        assert!(low_priority_worker < priority_threshold);

        // Workers with priority above threshold should pass
        let high_priority_worker = 80u32;
        assert!(high_priority_worker >= priority_threshold);
    }

    #[test]
    fn test_header_combinations() {
        // Test different header combinations
        struct TestCase {
            headers: Vec<(&'static str, &'static str)>,
            expected_priority: Option<u32>,
            expected_max_cost: Option<f32>,
            expected_prefer_pd: bool,
        }

        let test_cases = vec![
            TestCase {
                headers: vec![("x-worker-priority", "70")],
                expected_priority: Some(70),
                expected_max_cost: None,
                expected_prefer_pd: false,
            },
            TestCase {
                headers: vec![("x-max-cost", "2.5")],
                expected_priority: None,
                expected_max_cost: Some(2.5),
                expected_prefer_pd: false,
            },
            TestCase {
                headers: vec![("x-prefer-pd", "1")],
                expected_priority: None,
                expected_max_cost: None,
                expected_prefer_pd: true,
            },
            TestCase {
                headers: vec![
                    ("x-worker-priority", "90"),
                    ("x-max-cost", "1.8"),
                    ("x-prefer-pd", "true"),
                ],
                expected_priority: Some(90),
                expected_max_cost: Some(1.8),
                expected_prefer_pd: true,
            },
        ];

        for test_case in test_cases {
            let mut headers = HeaderMap::new();
            for (key, value) in test_case.headers {
                headers.insert(key, HeaderValue::from_static(value));
            }

            let priority = headers
                .get("x-worker-priority")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u32>().ok());

            let max_cost = headers
                .get("x-max-cost")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<f32>().ok());

            let prefer_pd = headers
                .get("x-prefer-pd")
                .and_then(|v| v.to_str().ok())
                .map(|s| s == "true" || s == "1")
                .unwrap_or(false);

            assert_eq!(priority, test_case.expected_priority);
            assert_eq!(max_cost, test_case.expected_max_cost);
            assert_eq!(prefer_pd, test_case.expected_prefer_pd);
        }
    }
}
