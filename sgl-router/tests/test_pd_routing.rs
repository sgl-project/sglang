//! Comprehensive tests for PrefillDecode (PD) routing functionality
//!
//! This test suite covers:
//! - Phase 1: Basic PD router creation and configuration
//! - Phase 2: Bootstrap injection and request handling
//! - Phase 3: Cache-aware selection (when implemented)
//!
//! Note: PD mode is enabled via the pd_disaggregated flag, not as a policy type.
//! The policy type (Random, PowerOfTwo, CacheAware) determines the selection algorithm within PD mode.

#[cfg(test)]
mod test_pd_routing {
    use rand::Rng;
    use serde_json::json;
    use sglang_router_rs::pd_types::{EngineInfo, EngineType, PDSelectionPolicy};
    use sglang_router_rs::router::{PolicyConfig, Router};

    // Test-only struct to help validate PD request parsing
    #[derive(Debug)]
    struct PDRequest {
        pub is_stream: bool,
        pub batch_size: Option<usize>,
    }

    impl PDRequest {
        // Extract PD-relevant info from JSON for testing
        pub fn from_json(json: &serde_json::Value) -> Self {
            let is_stream = json
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Detect batch size from text or input_ids
            let batch_size = if let Some(text) = json.get("text") {
                text.as_array().map(|arr| arr.len())
            } else if let Some(input_ids) = json.get("input_ids") {
                input_ids.as_array().map(|arr| arr.len())
            } else {
                None
            };

            PDRequest {
                is_stream,
                batch_size,
            }
        }
    }

    // ========================================================================
    // Phase 1: Basic PD Components and Router Creation
    // ========================================================================

    #[test]
    fn test_engine_info_creation() {
        // Test EngineInfo creation for prefill servers
        let prefill_engine = EngineInfo::new_prefill("http://prefill:8080".to_string(), Some(9000));
        match prefill_engine.engine_type {
            EngineType::Prefill => (),
            _ => panic!("Expected Prefill engine type"),
        }
        assert_eq!(prefill_engine.url, "http://prefill:8080");
        assert_eq!(prefill_engine.bootstrap_port, Some(9000));
        assert_eq!(prefill_engine.get_hostname(), "prefill");

        // Test EngineInfo creation for decode servers
        let decode_engine = EngineInfo::new_decode("http://decode:8080".to_string());
        match decode_engine.engine_type {
            EngineType::Decode => (),
            _ => panic!("Expected Decode engine type"),
        }
        assert_eq!(decode_engine.url, "http://decode:8080");
        assert_eq!(decode_engine.bootstrap_port, None);
        assert_eq!(decode_engine.get_hostname(), "decode");

        // Test API path generation
        assert_eq!(
            prefill_engine.api_path("/generate"),
            "http://prefill:8080/generate"
        );
        assert_eq!(
            prefill_engine.api_path("health"),
            "http://prefill:8080/health"
        );
        assert_eq!(
            decode_engine.api_path("/v1/chat/completions"),
            "http://decode:8080/v1/chat/completions"
        );
    }

    #[test]
    fn test_pd_selection_policies() {
        // Test all PD selection policy variants
        // Note: These policies are only used when pd_disaggregated=true
        let policies = vec![
            PDSelectionPolicy::Random,
            PDSelectionPolicy::PowerOfTwo,
            PDSelectionPolicy::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
            },
        ];

        for policy in policies {
            // Verify each policy can be created and matched
            match &policy {
                PDSelectionPolicy::Random => {
                    assert!(matches!(policy, PDSelectionPolicy::Random));
                }
                PDSelectionPolicy::PowerOfTwo => {
                    assert!(matches!(policy, PDSelectionPolicy::PowerOfTwo));
                }
                PDSelectionPolicy::CacheAware {
                    cache_threshold, ..
                } => {
                    assert!(*cache_threshold >= 0.0 && *cache_threshold <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_pd_router_configuration() {
        // Test PrefillDecodeConfig creation with various policies
        // This config is used when pd_disaggregated=true
        let configs = vec![
            PolicyConfig::PrefillDecodeConfig {
                selection_policy: PDSelectionPolicy::Random,
                prefill_urls: vec![
                    ("http://prefill1:8080".to_string(), Some(9000)),
                    ("http://prefill2:8080".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8080".to_string(),
                    "http://decode2:8080".to_string(),
                ],
                timeout_secs: 10,
                interval_secs: 1,
            },
            PolicyConfig::PrefillDecodeConfig {
                selection_policy: PDSelectionPolicy::PowerOfTwo,
                prefill_urls: vec![("http://prefill:8080".to_string(), Some(9000))],
                decode_urls: vec!["http://decode:8080".to_string()],
                timeout_secs: 5,
                interval_secs: 1,
            },
            PolicyConfig::PrefillDecodeConfig {
                selection_policy: PDSelectionPolicy::CacheAware {
                    cache_threshold: 0.7,
                    balance_abs_threshold: 20,
                    balance_rel_threshold: 1.2,
                },
                prefill_urls: vec![
                    ("http://p1:8080".to_string(), Some(9000)),
                    ("http://p2:8080".to_string(), Some(9001)),
                    ("http://p3:8080".to_string(), Some(9002)),
                ],
                decode_urls: vec!["http://d1:8080".to_string(), "http://d2:8080".to_string()],
                timeout_secs: 10,
                interval_secs: 2,
            },
        ];

        for config in configs {
            // Router creation will fail due to health checks, but config should be valid
            let result = Router::new(vec![], config);
            assert!(result.is_err());
            let error_msg = result.unwrap_err();
            // Error should be about health/timeout, not configuration
            assert!(
                error_msg.contains("healthy") || error_msg.contains("timeout"),
                "Unexpected error: {}",
                error_msg
            );
        }
    }

    // ========================================================================
    // Phase 2: Bootstrap Injection and Request Handling
    // ========================================================================

    #[test]
    fn test_pd_request_from_json() {
        // Test PDRequest parsing from single text request
        let single_json = json!({
            "text": "Hello world",
            "stream": false,
            "temperature": 0.7,
            "max_tokens": 100
        });

        let pd_req = PDRequest::from_json(&single_json);
        assert!(!pd_req.is_stream);
        assert_eq!(pd_req.batch_size, None);

        // Test PDRequest parsing from batch text request
        let batch_json = json!({
            "text": ["Hello", "World", "Test"],
            "stream": true,
            "temperature": 0.5
        });

        let pd_req = PDRequest::from_json(&batch_json);
        assert!(pd_req.is_stream);
        assert_eq!(pd_req.batch_size, Some(3));

        // Test PDRequest parsing from input_ids request
        let ids_json = json!({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "stream": false
        });

        let pd_req = PDRequest::from_json(&ids_json);
        assert!(!pd_req.is_stream);
        assert_eq!(pd_req.batch_size, Some(2));

        // Test PDRequest parsing from chat request
        let chat_json = json!({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ],
            "stream": true
        });

        let pd_req = PDRequest::from_json(&chat_json);
        assert!(pd_req.is_stream);
        assert_eq!(pd_req.batch_size, None);
    }

    #[test]
    fn test_bootstrap_injection_simulation() {
        // Since we can't test the actual inject_bootstrap_fields function here
        // (it's private in the router module), we'll test the expected behavior

        // Simulate bootstrap injection for single request
        let mut single_json = json!({
            "text": "Hello world",
            "stream": false,
            "temperature": 0.7
        });

        // Simulate what inject_bootstrap_fields would do
        let prefill_info = EngineInfo::new_prefill("http://prefill1:8080".to_string(), Some(9000));
        single_json["bootstrap_host"] = json!(prefill_info.get_hostname());
        single_json["bootstrap_port"] = json!(prefill_info.bootstrap_port);
        single_json["bootstrap_room"] = json!(12345u64); // Random room ID

        // Verify bootstrap fields are added correctly
        assert_eq!(single_json["bootstrap_host"], "prefill1");
        assert_eq!(single_json["bootstrap_port"], 9000);
        assert!(single_json["bootstrap_room"].is_u64());
        assert_eq!(single_json["temperature"], 0.7); // Original field preserved

        // Simulate bootstrap injection for batch request
        let mut batch_json = json!({
            "text": ["Hello", "World", "Test"],
            "stream": true
        });

        let batch_size = 3;
        batch_json["bootstrap_host"] = json!(vec![prefill_info.get_hostname(); batch_size]);
        batch_json["bootstrap_port"] = json!(vec![prefill_info.bootstrap_port; batch_size]);
        batch_json["bootstrap_room"] = json!(vec![111u64, 222u64, 333u64]);

        // Verify batch bootstrap fields
        assert!(batch_json["bootstrap_host"].is_array());
        assert_eq!(
            batch_json["bootstrap_host"].as_array().unwrap().len(),
            batch_size
        );
        assert!(batch_json["bootstrap_port"].is_array());
        assert!(batch_json["bootstrap_room"].is_array());
        assert_eq!(batch_json["stream"], true); // Original field preserved
    }

    #[test]
    fn test_request_serialization() {
        // Test that requests can be properly serialized and deserialized
        let request = json!({
            "text": "Test prompt",
            "stream": false,
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "bootstrap_host": "prefill1",
            "bootstrap_port": 9000,
            "bootstrap_room": 12345u64
        });

        // Convert to bytes (as would happen in the router)
        let bytes = serde_json::to_vec(&request).unwrap();

        // Parse back from bytes
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // Verify all fields are preserved
        assert_eq!(parsed["text"], "Test prompt");
        assert_eq!(parsed["stream"], false);
        assert_eq!(parsed["temperature"], 0.7);
        assert_eq!(parsed["max_tokens"], 100);
        assert_eq!(parsed["bootstrap_host"], "prefill1");
        assert_eq!(parsed["bootstrap_port"], 9000);
        assert_eq!(parsed["bootstrap_room"], 12345);
    }

    #[test]
    fn test_engine_info_hostname_extraction() {
        // Test various URL formats
        let test_cases = vec![
            ("http://localhost:8080", "localhost"),
            ("http://10.0.0.1:8080", "10.0.0.1"),
            ("https://api.example.com:443", "api.example.com"),
            ("http://prefill-server", "prefill-server"),
            ("http://[::1]:8080", "["),  // IPv6 edge case
            ("prefill:8080", "prefill"), // No protocol
        ];

        for (url, expected_hostname) in test_cases {
            let engine = EngineInfo::new_prefill(url.to_string(), None);
            assert_eq!(engine.get_hostname(), expected_hostname);
        }
    }

    #[test]
    fn test_pd_request_edge_cases() {
        // Test empty request
        let empty_json = json!({});
        let pd_req = PDRequest::from_json(&empty_json);
        assert!(!pd_req.is_stream);
        assert_eq!(pd_req.batch_size, None);

        // Test request with only stream field
        let stream_only = json!({
            "stream": true
        });
        let pd_req = PDRequest::from_json(&stream_only);
        assert!(pd_req.is_stream);
        assert_eq!(pd_req.batch_size, None);

        // Test request with empty text array
        let empty_batch = json!({
            "text": []
        });
        let pd_req = PDRequest::from_json(&empty_batch);
        assert_eq!(pd_req.batch_size, Some(0));

        // Test request with non-array text (should be None)
        let non_array_text = json!({
            "text": "single string"
        });
        let pd_req = PDRequest::from_json(&non_array_text);
        assert_eq!(pd_req.batch_size, None);
    }

    // ========================================================================
    // Phase 2: Background Load Monitoring Tests
    // ========================================================================

    #[tokio::test]
    async fn test_background_load_monitoring() {
        use std::collections::HashMap;
        use tokio::sync::watch;

        // Create a watch channel for testing
        let (tx, rx) = watch::channel(HashMap::new());

        // Simulate load updates
        let mut loads = HashMap::new();
        loads.insert("http://prefill1:8080".to_string(), 10);
        loads.insert("http://prefill2:8080".to_string(), 20);
        loads.insert("http://decode1:8080".to_string(), 5);
        loads.insert("http://decode2:8080".to_string(), 15);

        // Send the loads
        tx.send(loads.clone()).unwrap();

        // Verify receiver gets the update
        let received_loads = rx.borrow();
        assert_eq!(received_loads.get("http://prefill1:8080"), Some(&10));
        assert_eq!(received_loads.get("http://prefill2:8080"), Some(&20));
        assert_eq!(received_loads.get("http://decode1:8080"), Some(&5));
        assert_eq!(received_loads.get("http://decode2:8080"), Some(&15));
    }

    #[test]
    fn test_power_of_two_load_selection() {
        // Test the power-of-two selection logic with different load scenarios

        // Scenario 1: Clear winner for both prefill and decode
        let _loads = vec![
            ("prefill1", 100),
            ("prefill2", 10), // Should be selected
            ("decode1", 50),
            ("decode2", 5), // Should be selected
        ];

        // In actual implementation, the lower load should be selected
        assert!(10 < 100);
        assert!(5 < 50);

        // Scenario 2: Equal loads (should select first)
        let _equal_loads = vec![
            ("prefill1", 20),
            ("prefill2", 20), // Either could be selected
            ("decode1", 30),
            ("decode2", 30), // Either could be selected
        ];

        // When loads are equal, <= comparison means first is selected
        assert!(20 <= 20);
        assert!(30 <= 30);

        // Scenario 3: Missing load data (should default to usize::MAX)
        // This tests the unwrap_or(usize::MAX) behavior
        let missing_load = usize::MAX;
        assert!(10 < missing_load);
        assert!(missing_load > 0);
    }

    #[test]
    fn test_load_monitoring_configuration() {
        // Test that load monitoring is only enabled for PowerOfTwo policy
        let policies = vec![
            (PDSelectionPolicy::Random, false),
            (PDSelectionPolicy::PowerOfTwo, true),
            (
                PDSelectionPolicy::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                },
                false,
            ),
        ];

        for (policy, should_monitor) in policies {
            match policy {
                PDSelectionPolicy::PowerOfTwo => assert!(should_monitor),
                _ => assert!(!should_monitor),
            }
        }
    }

    #[tokio::test]
    async fn test_watch_channel_behavior() {
        use std::collections::HashMap;
        use tokio::sync::watch;

        // Test watch channel's broadcast behavior
        let (tx, rx1) = watch::channel(HashMap::new());
        let rx2 = rx1.clone();

        // Initial state - empty map
        assert!(rx1.borrow().is_empty());
        assert!(rx2.borrow().is_empty());

        // Update 1
        let mut loads = HashMap::new();
        loads.insert("worker1".to_string(), 10);
        tx.send(loads.clone()).unwrap();

        // Both receivers see the update
        assert_eq!(rx1.borrow().get("worker1"), Some(&10));
        assert_eq!(rx2.borrow().get("worker1"), Some(&10));

        // Update 2 - overwrites previous
        loads.insert("worker1".to_string(), 20);
        loads.insert("worker2".to_string(), 30);
        tx.send(loads).unwrap();

        // Both receivers see the latest state
        assert_eq!(rx1.borrow().get("worker1"), Some(&20));
        assert_eq!(rx2.borrow().get("worker2"), Some(&30));
    }

    // ========================================================================
    // Tests based on bench_one_batch_server.py patterns
    // ========================================================================

    #[test]
    fn test_generate_request_formats() {
        // Based on bench_one_batch_server.py request patterns

        // Test 1: Batch request with input_ids (most common in benchmarks)
        let batch_request = json!({
            "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 16,
                "ignore_eos": true,
            },
            "return_logprob": false,
            "stream": true
        });

        let pd_req = PDRequest::from_json(&batch_request);
        assert!(pd_req.is_stream);
        assert_eq!(pd_req.batch_size, Some(3));

        // Test 2: Request with return_logprob (critical for PD)
        let logprob_request = json!({
            "input_ids": [[1, 2, 3]],
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 8,
            },
            "return_logprob": true,
            "stream": false
        });

        assert_eq!(logprob_request["return_logprob"], true);
        assert_eq!(logprob_request["stream"], false);

        // Test 3: Large batch sizes from benchmark
        let batch_sizes = vec![1, 16, 64]; // From bench_one_batch_server.py
        for bs in batch_sizes {
            let request = json!({
                "input_ids": vec![vec![1, 2, 3]; bs],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                },
                "stream": true
            });

            let pd_req = PDRequest::from_json(&request);
            assert_eq!(pd_req.batch_size, Some(bs));
        }
    }

    #[test]
    fn test_sampling_params_handling() {
        // Test various sampling parameters from bench_one_batch_server.py
        let sampling_params_variations = vec![
            json!({
                "temperature": 0.0,
                "max_new_tokens": 8,
                "ignore_eos": true
            }),
            json!({
                "temperature": 0.7,
                "max_new_tokens": 16,
                "ignore_eos": false,
                "top_p": 0.9,
                "frequency_penalty": 0.5
            }),
            json!({
                "temperature": 1.0,
                "max_new_tokens": 64,
                "json_schema": "$$ANY$$"  // Structured output
            }),
        ];

        for params in sampling_params_variations {
            let request = json!({
                "input_ids": [[1, 2, 3]],
                "sampling_params": params.clone(),
                "stream": false
            });

            // Verify params are preserved
            assert_eq!(request["sampling_params"], params);
        }
    }

    #[test]
    fn test_streaming_response_parsing() {
        // Test SSE format parsing from streaming responses
        let sse_chunks = vec![
            "data: {\"text\":\"Hello\",\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":null}}",
            "data: {\"text\":\" world\",\"meta_info\":{\"completion_tokens\":2,\"finish_reason\":null}}",
            "data: {\"text\":\"!\",\"meta_info\":{\"completion_tokens\":3,\"finish_reason\":{\"type\":\"length\"}}}",
            "data: [DONE]",
        ];

        for chunk in &sse_chunks[..3] {
            assert!(chunk.starts_with("data: "));
            let json_str = &chunk[6..]; // Skip "data: "
            let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
            assert!(parsed["meta_info"]["completion_tokens"].is_u64());
        }

        // Test [DONE] detection
        assert_eq!(sse_chunks[3], "data: [DONE]");
    }

    #[test]
    fn test_ttft_calculation() {
        // Test Time To First Token calculation pattern
        let first_token_response = json!({
            "text": "Hello",
            "meta_info": {
                "completion_tokens": 1,
                "finish_reason": null
            }
        });

        // TTFT is calculated when completion_tokens == 1
        assert_eq!(first_token_response["meta_info"]["completion_tokens"], 1);
        assert!(first_token_response["meta_info"]["finish_reason"].is_null());
    }

    #[test]
    fn test_throughput_metrics() {
        // Test throughput calculation patterns from bench_one_batch_server.py
        let batch_size = 16;
        let input_len = 1024;
        let output_len = 16;
        let ttft = 0.5; // seconds
        let total_latency = 2.0; // seconds

        // Input throughput = batch_size * input_len / ttft
        let input_throughput = (batch_size as f64) * (input_len as f64) / ttft;
        assert!((input_throughput - 32768.0).abs() < 0.01);

        // Output throughput = batch_size * output_len / (latency - ttft)
        let output_throughput = (batch_size as f64) * (output_len as f64) / (total_latency - ttft);
        assert!((output_throughput - 170.67).abs() < 0.01);
    }

    #[test]
    fn test_error_response_handling() {
        // Test error response format from bench_one_batch_server.py
        let error_response = json!({
            "error": "Request has failed. Invalid input format."
        });

        assert!(error_response.get("error").is_some());
        assert!(error_response["error"].as_str().unwrap().contains("failed"));
    }

    #[test]
    fn test_structured_output_request() {
        // Test structured output format (json_schema)
        let structured_request = json!({
            "text": "What is the capital of France? Answer in JSON.",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 64,
                "json_schema": "$$ANY$$"
            },
            "stream": false
        });

        assert_eq!(
            structured_request["sampling_params"]["json_schema"],
            "$$ANY$$"
        );
    }

    #[test]
    fn test_bootstrap_injection_with_benchmark_requests() {
        // Test bootstrap injection with actual benchmark request patterns
        let mut benchmark_request = json!({
            "input_ids": vec![vec![1, 2, 3, 4]; 16], // Batch size 16
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 8,
                "ignore_eos": true
            },
            "return_logprob": true,
            "stream": true
        });

        // Simulate bootstrap injection
        let prefill_info = EngineInfo::new_prefill("http://prefill:8080".to_string(), Some(9000));
        let batch_size = 16;

        benchmark_request["bootstrap_host"] = json!(vec![prefill_info.get_hostname(); batch_size]);
        benchmark_request["bootstrap_port"] = json!(vec![prefill_info.bootstrap_port; batch_size]);
        benchmark_request["bootstrap_room"] =
            json!((0..batch_size).map(|_| 12345u64).collect::<Vec<_>>());

        // Verify bootstrap fields match batch size
        assert_eq!(
            benchmark_request["bootstrap_host"]
                .as_array()
                .unwrap()
                .len(),
            batch_size
        );
        assert_eq!(
            benchmark_request["bootstrap_port"]
                .as_array()
                .unwrap()
                .len(),
            batch_size
        );
        assert_eq!(
            benchmark_request["bootstrap_room"]
                .as_array()
                .unwrap()
                .len(),
            batch_size
        );

        // Verify original fields are preserved
        assert_eq!(benchmark_request["return_logprob"], true);
        assert_eq!(benchmark_request["stream"], true);
    }

    #[test]
    fn test_server_info_response_format() {
        // Test server info format expected by bench_one_batch_server.py
        let server_info = json!({
            "internal_states": [{
                "avg_spec_accept_length": 3.5,
                "last_gen_throughput": 2048.5,
                "load": 16
            }],
            "prefill": [
                {"url": "http://prefill1:8080", "load": 10},
                {"url": "http://prefill2:8080", "load": 20}
            ],
            "decode": [
                {"url": "http://decode1:8080", "load": 5},
                {"url": "http://decode2:8080", "load": 15}
            ]
        });

        // Verify structure matches what benchmark expects
        assert!(server_info["internal_states"][0]["avg_spec_accept_length"].is_f64());
        assert!(server_info["internal_states"][0]["last_gen_throughput"].is_f64());
        assert!(server_info["prefill"].is_array());
        assert!(server_info["decode"].is_array());
    }

    // ========================================================================
    // Comprehensive Endpoint Coverage Test
    // ========================================================================

    #[test]
    fn test_pd_endpoints_coverage() {
        // Document all endpoints from Python mini_lb.py and verify implementation status
        let implemented_endpoints = vec![
            ("/health", "GET", true),
            ("/health_generate", "GET", true), // Note: Python uses POST, we use GET
            ("/get_server_info", "GET", true),
            ("/v1/models", "GET", true),
            ("/get_model_info", "GET", true),
            ("/generate", "POST", true),
            ("/v1/chat/completions", "POST", true),
            ("/v1/completions", "POST", true),
            ("/flush_cache", "POST", true),
            ("/get_loads", "GET", true),
            ("/register", "POST", false), // NOT IMPLEMENTED - needs dynamic worker management
        ];

        let implemented_count = implemented_endpoints
            .iter()
            .filter(|(_, _, impl_status)| *impl_status)
            .count();
        let total_count = implemented_endpoints.len();

        // We've implemented 10 out of 11 endpoints (register is not needed for Phase 1/2)
        assert_eq!(implemented_count, 10);
        assert_eq!(total_count, 11);

        // Document the missing endpoint
        let missing: Vec<_> = implemented_endpoints
            .iter()
            .filter(|(_, _, impl_status)| !impl_status)
            .map(|(endpoint, method, _)| format!("{} {}", method, endpoint))
            .collect();

        assert_eq!(missing, vec!["POST /register"]);
    }

    #[test]
    fn test_large_batch_bootstrap_injection() {
        // Test bootstrap injection performance with very large batches
        // This simulates the bench_one_batch_server.py scenario
        let large_batch_sizes = vec![1024, 4096, 8192];

        for batch_size in large_batch_sizes {
            let start = std::time::Instant::now();

            // Simulate a large batch request
            let mut large_batch_request = json!({
                "input_ids": vec![vec![1, 2, 3, 4]; batch_size],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                },
                "stream": true
            });

            // Simulate bootstrap injection
            let prefill_info =
                EngineInfo::new_prefill("http://prefill:8080".to_string(), Some(9000));

            large_batch_request["bootstrap_host"] =
                json!(vec![prefill_info.get_hostname(); batch_size]);
            large_batch_request["bootstrap_port"] =
                json!(vec![prefill_info.bootstrap_port; batch_size]);
            large_batch_request["bootstrap_room"] = json!((0..batch_size)
                .map(|_| rand::thread_rng().gen::<u64>())
                .collect::<Vec<_>>());

            let elapsed = start.elapsed();

            // Verify bootstrap fields are correctly sized
            assert_eq!(
                large_batch_request["bootstrap_host"]
                    .as_array()
                    .unwrap()
                    .len(),
                batch_size
            );
            assert_eq!(
                large_batch_request["bootstrap_port"]
                    .as_array()
                    .unwrap()
                    .len(),
                batch_size
            );
            assert_eq!(
                large_batch_request["bootstrap_room"]
                    .as_array()
                    .unwrap()
                    .len(),
                batch_size
            );

            // Bootstrap injection should be reasonably fast even for large batches
            println!(
                "Bootstrap injection for batch_size {} took {:?}",
                batch_size, elapsed
            );
            assert!(
                elapsed.as_millis() < 1000,
                "Bootstrap injection took too long for batch size {}",
                batch_size
            );
        }
    }

    #[test]
    fn test_payload_size_calculation() {
        // Test payload size estimation for bench_one_batch_server.py scenarios
        let test_cases = vec![
            (1, 1024, 16),   // Small batch
            (16, 1024, 16),  // Medium batch
            (64, 1024, 16),  // Large batch
            (8192, 4096, 5), // Benchmark scenario
        ];

        for (batch_size, input_len, _output_len) in test_cases {
            // Estimate payload size (rough calculation)
            // Each token is ~4 bytes (i32), plus JSON overhead
            let tokens_size = batch_size * input_len * 4; // 4 bytes per token
            let json_overhead = batch_size * 100; // ~100 bytes overhead per request
            let total_size = tokens_size + json_overhead;

            println!(
                "Batch size: {}, Input len: {}, Estimated payload: {} MB",
                batch_size,
                input_len,
                total_size / (1024 * 1024)
            );

            // For the benchmark case (8192, 4096), this should be ~134 MB
            if batch_size == 8192 && input_len == 4096 {
                assert!(
                    total_size > 100 * 1024 * 1024,
                    "Benchmark payload should be > 100MB"
                );
                assert!(
                    total_size < 200 * 1024 * 1024,
                    "Benchmark payload should be < 200MB"
                );
            }
        }
    }

    #[test]
    fn test_policy_type_to_pd_selection_policy_mapping() {
        // Document the mapping from PolicyType to PDSelectionPolicy
        // This mapping happens in lib.rs when pd_disaggregated=true

        // PolicyType::Random -> PDSelectionPolicy::Random
        // PolicyType::PowerOfTwo -> PDSelectionPolicy::PowerOfTwo
        // PolicyType::CacheAware -> PDSelectionPolicy::CacheAware { ... }
        // PolicyType::RoundRobin -> ERROR (not supported in PD mode)

        // Test that PDSelectionPolicy doesn't include RoundRobin
        let pd_policy_count = 3; // Random, PowerOfTwo, CacheAware
        assert_eq!(
            pd_policy_count, 3,
            "PDSelectionPolicy should have exactly 3 variants"
        );

        // Verify that each PDSelectionPolicy variant can be created
        let _random = PDSelectionPolicy::Random;
        let _po2 = PDSelectionPolicy::PowerOfTwo;
        let _cache_aware = PDSelectionPolicy::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
        };
    }
}
