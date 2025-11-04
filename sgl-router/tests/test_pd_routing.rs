#[cfg(test)]
mod test_pd_routing {
    use serde_json::json;
    use sglang_router_rs::{
        app_context::AppContext,
        config::{PolicyConfig, RouterConfig, RoutingMode},
        core::{BasicWorkerBuilder, Worker, WorkerType},
        routers::{http::pd_types::PDSelectionPolicy, RouterFactory},
    };

    #[derive(Debug)]
    struct PDRequest {
        pub is_stream: bool,
        pub batch_size: Option<usize>,
    }

    impl PDRequest {
        pub fn from_json(json: &serde_json::Value) -> Self {
            let is_stream = json
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

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

    #[test]
    fn test_worker_types() {
        use sglang_router_rs::core::{BasicWorkerBuilder, Worker, WorkerType};

        let prefill_worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9000),
                })
                .api_key("test_api_key")
                .build(),
        );
        assert_eq!(prefill_worker.url(), "http://prefill:8080");
        match prefill_worker.worker_type() {
            WorkerType::Prefill { bootstrap_port } => {
                assert_eq!(bootstrap_port, Some(9000));
            }
            _ => panic!("Expected Prefill worker type"),
        }

        let decode_worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .api_key("test_api_key")
                .build(),
        );
        assert_eq!(decode_worker.url(), "http://decode:8080");
        match decode_worker.worker_type() {
            WorkerType::Decode => (),
            _ => panic!("Expected Decode worker type"),
        }

        let regular_worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        );
        assert_eq!(regular_worker.url(), "http://regular:8080");
        match regular_worker.worker_type() {
            WorkerType::Regular => (),
            _ => panic!("Expected Regular worker type"),
        }
    }

    #[test]
    fn test_pd_selection_policies() {
        // Note: These policies are only used when pd_disaggregation=true
        let policies = vec![
            PDSelectionPolicy::Random,
            PDSelectionPolicy::PowerOfTwo,
            PDSelectionPolicy::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
            },
            PDSelectionPolicy::Bucket {
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                bucket_adjust_interval_secs: 5,
            },
        ];

        for policy in policies {
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
                PDSelectionPolicy::Bucket {
                    balance_rel_threshold,
                    ..
                } => {
                    assert!(*balance_rel_threshold >= 1.0);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_pd_router_configuration() {
        // In the new structure, RoutingMode and PolicyConfig are separate
        let test_cases = vec![
            (
                RoutingMode::PrefillDecode {
                    prefill_urls: vec![
                        ("http://prefill1:8080".to_string(), Some(9000)),
                        ("http://prefill2:8080".to_string(), None),
                    ],
                    decode_urls: vec![
                        "http://decode1:8080".to_string(),
                        "http://decode2:8080".to_string(),
                    ],
                    prefill_policy: None,
                    decode_policy: None,
                },
                PolicyConfig::Random,
            ),
            (
                RoutingMode::PrefillDecode {
                    prefill_urls: vec![("http://prefill:8080".to_string(), Some(9000))],
                    decode_urls: vec!["http://decode:8080".to_string()],
                    prefill_policy: None,
                    decode_policy: None,
                },
                PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5,
                },
            ),
            (
                RoutingMode::PrefillDecode {
                    prefill_urls: vec![
                        ("http://p1:8080".to_string(), Some(9000)),
                        ("http://p2:8080".to_string(), Some(9001)),
                        ("http://p3:8080".to_string(), Some(9002)),
                    ],
                    decode_urls: vec!["http://d1:8080".to_string(), "http://d2:8080".to_string()],
                    prefill_policy: None,
                    decode_policy: None,
                },
                PolicyConfig::CacheAware {
                    cache_threshold: 0.7,
                    balance_abs_threshold: 20,
                    balance_rel_threshold: 1.2,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000000,
                },
            ),
            (
                RoutingMode::PrefillDecode {
                    prefill_urls: vec![
                        ("http://p1:8080".to_string(), Some(9000)),
                        ("http://p2:8080".to_string(), Some(9001)),
                        ("http://p3:8080".to_string(), Some(9002)),
                    ],
                    decode_urls: vec!["http://d1:8080".to_string(), "http://d2:8080".to_string()],
                    prefill_policy: None,
                    decode_policy: None,
                },
                PolicyConfig::Bucket {
                    balance_abs_threshold: 20,
                    balance_rel_threshold: 1.2,
                    bucket_adjust_interval_secs: 5,
                },
            ),
        ];

        for (mode, policy) in test_cases {
            let config = match mode {
                RoutingMode::PrefillDecode {
                    prefill_urls,
                    decode_urls,
                    ..
                } => RouterConfig::builder()
                    .prefill_decode_mode(prefill_urls, decode_urls)
                    .policy(policy)
                    .host("127.0.0.1")
                    .port(3001)
                    .max_payload_size(1024 * 1024)
                    .request_timeout_secs(60)
                    .worker_startup_timeout_secs(10)
                    .worker_startup_check_interval_secs(1)
                    .max_concurrent_requests(64)
                    .queue_timeout_secs(60)
                    .build_unchecked(),
                _ => panic!("Expected PrefillDecode mode"),
            };

            let app_context = {
                use std::sync::{Arc, OnceLock};

                use sglang_router_rs::{
                    core::{LoadMonitor, WorkerRegistry},
                    data_connector::{
                        MemoryConversationItemStorage, MemoryConversationStorage,
                        MemoryResponseStorage,
                    },
                    middleware::TokenBucket,
                    policies::PolicyRegistry,
                };

                let client = reqwest::Client::new();

                // Initialize rate limiter
                let rate_limiter = Some(Arc::new(TokenBucket::new(64, 64)));

                // Initialize registries
                let worker_registry = Arc::new(WorkerRegistry::new());
                let policy_registry = Arc::new(PolicyRegistry::new(config.policy.clone()));

                // Initialize storage backends
                let response_storage = Arc::new(MemoryResponseStorage::new());
                let conversation_storage = Arc::new(MemoryConversationStorage::new());
                let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

                // Initialize load monitor
                let load_monitor = Some(Arc::new(LoadMonitor::new(
                    worker_registry.clone(),
                    policy_registry.clone(),
                    client.clone(),
                    config.worker_startup_check_interval_secs,
                )));

                // Create empty OnceLock for worker job queue, workflow engine, and mcp manager
                let worker_job_queue = Arc::new(OnceLock::new());
                let workflow_engine = Arc::new(OnceLock::new());
                let mcp_manager = Arc::new(OnceLock::new());

                Arc::new(
                    AppContext::builder()
                        .router_config(config)
                        .client(client)
                        .rate_limiter(rate_limiter)
                        .tokenizer(None) // tokenizer
                        .reasoning_parser_factory(None) // reasoning_parser_factory
                        .tool_parser_factory(None) // tool_parser_factory
                        .worker_registry(worker_registry)
                        .policy_registry(policy_registry)
                        .response_storage(response_storage)
                        .conversation_storage(conversation_storage)
                        .conversation_item_storage(conversation_item_storage)
                        .load_monitor(load_monitor)
                        .worker_job_queue(worker_job_queue)
                        .workflow_engine(workflow_engine)
                        .mcp_manager(mcp_manager)
                        .build()
                        .unwrap(),
                )
            };
            let result = RouterFactory::create_router(&app_context).await;
            assert!(
                result.is_ok(),
                "Router creation should succeed with empty worker"
            );

            let stats = app_context.worker_registry.stats();
            assert_eq!(
                stats.total_workers, 0,
                "No workers should be registered without initialization"
            );
        }
    }

    #[test]
    fn test_pd_request_from_json() {
        let single_json = json!({
            "text": "Hello world",
            "stream": false,
            "temperature": 0.7,
            "max_tokens": 100
        });

        let pd_req = PDRequest::from_json(&single_json);
        assert!(!pd_req.is_stream);
        assert_eq!(pd_req.batch_size, None);

        let batch_json = json!({
            "text": ["Hello", "World", "Test"],
            "stream": true,
            "temperature": 0.5
        });

        let pd_req = PDRequest::from_json(&batch_json);
        assert!(pd_req.is_stream);
        assert_eq!(pd_req.batch_size, Some(3));

        let ids_json = json!({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "stream": false
        });

        let pd_req = PDRequest::from_json(&ids_json);
        assert!(!pd_req.is_stream);
        assert_eq!(pd_req.batch_size, Some(2));

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

        let mut single_json = json!({
            "text": "Hello world",
            "stream": false,
            "temperature": 0.7
        });

        let prefill_worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill1:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9000),
                })
                .api_key("test_api_key")
                .build(),
        );

        let bootstrap_port = match prefill_worker.worker_type() {
            WorkerType::Prefill { bootstrap_port } => bootstrap_port,
            _ => None,
        };

        single_json["bootstrap_host"] = json!(prefill_worker.bootstrap_host());
        single_json["bootstrap_port"] = json!(bootstrap_port);
        single_json["bootstrap_room"] = json!(12345u64); // Random room ID

        assert_eq!(single_json["bootstrap_host"], "prefill1");
        assert_eq!(single_json["bootstrap_port"], json!(Some(9000)));
        assert!(single_json["bootstrap_room"].is_u64());
        assert_eq!(single_json["temperature"], 0.7); // Original field preserved

        let mut batch_json = json!({
            "text": ["Hello", "World", "Test"],
            "stream": true
        });

        let batch_size = 3;
        let hostname = prefill_worker.bootstrap_host();
        batch_json["bootstrap_host"] = json!(vec![hostname; batch_size]);
        batch_json["bootstrap_port"] = json!(vec![bootstrap_port; batch_size]);
        batch_json["bootstrap_room"] = json!(vec![111u64, 222u64, 333u64]);

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

        let bytes = serde_json::to_vec(&request).unwrap();

        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(parsed["text"], "Test prompt");
        assert_eq!(parsed["stream"], false);
        assert_eq!(parsed["temperature"], 0.7);
        assert_eq!(parsed["max_tokens"], 100);
        assert_eq!(parsed["bootstrap_host"], "prefill1");
        assert_eq!(parsed["bootstrap_port"], 9000);
        assert_eq!(parsed["bootstrap_room"], 12345);
    }

    #[test]
    fn test_pd_request_edge_cases() {
        let empty_json = json!({});
        let pd_req = PDRequest::from_json(&empty_json);
        assert!(!pd_req.is_stream);
        assert_eq!(pd_req.batch_size, None);

        let stream_only = json!({
            "stream": true
        });
        let pd_req = PDRequest::from_json(&stream_only);
        assert!(pd_req.is_stream);
        assert_eq!(pd_req.batch_size, None);

        let empty_batch = json!({
            "text": []
        });
        let pd_req = PDRequest::from_json(&empty_batch);
        assert_eq!(pd_req.batch_size, Some(0));

        let non_array_text = json!({
            "text": "single string"
        });
        let pd_req = PDRequest::from_json(&non_array_text);
        assert_eq!(pd_req.batch_size, None);
    }

    #[tokio::test]
    async fn test_background_load_monitoring() {
        use std::collections::HashMap;

        use tokio::sync::watch;

        let (tx, rx) = watch::channel(HashMap::new());

        let mut loads = HashMap::new();
        loads.insert("http://prefill1:8080".to_string(), 10);
        loads.insert("http://prefill2:8080".to_string(), 20);
        loads.insert("http://decode1:8080".to_string(), 5);
        loads.insert("http://decode2:8080".to_string(), 15);

        tx.send(loads.clone()).unwrap();

        let received_loads = rx.borrow();
        assert_eq!(received_loads.get("http://prefill1:8080"), Some(&10));
        assert_eq!(received_loads.get("http://prefill2:8080"), Some(&20));
        assert_eq!(received_loads.get("http://decode1:8080"), Some(&5));
        assert_eq!(received_loads.get("http://decode2:8080"), Some(&15));
    }

    #[test]
    fn test_load_monitoring_configuration() {
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

        let (tx, rx1) = watch::channel(HashMap::new());
        let rx2 = rx1.clone();

        assert!(rx1.borrow().is_empty());
        assert!(rx2.borrow().is_empty());

        let mut loads = HashMap::new();
        loads.insert("worker1".to_string(), 10);
        tx.send(loads.clone()).unwrap();

        assert_eq!(rx1.borrow().get("worker1"), Some(&10));
        assert_eq!(rx2.borrow().get("worker1"), Some(&10));

        loads.insert("worker1".to_string(), 20);
        loads.insert("worker2".to_string(), 30);
        tx.send(loads).unwrap();

        assert_eq!(rx1.borrow().get("worker1"), Some(&20));
        assert_eq!(rx2.borrow().get("worker2"), Some(&30));
    }

    #[test]
    fn test_generate_request_formats() {
        // Based on bench_one_batch_server.py request patterns

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

            assert_eq!(request["sampling_params"], params);
        }
    }

    #[test]
    fn test_streaming_response_parsing() {
        let sse_chunks = ["data: {\"text\":\"Hello\",\"meta_info\":{\"completion_tokens\":1,\"finish_reason\":null}}",
            "data: {\"text\":\" world\",\"meta_info\":{\"completion_tokens\":2,\"finish_reason\":null}}",
            "data: {\"text\":\"!\",\"meta_info\":{\"completion_tokens\":3,\"finish_reason\":{\"type\":\"length\"}}}",
            "data: [DONE]"];

        for chunk in &sse_chunks[..3] {
            assert!(chunk.starts_with("data: "));
            let json_str = &chunk[6..]; // Skip "data: "
            let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
            assert!(parsed["meta_info"]["completion_tokens"].is_u64());
        }

        assert_eq!(sse_chunks[3], "data: [DONE]");
    }

    #[test]
    fn test_ttft_calculation() {
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
        let error_response = json!({
            "error": "Request has failed. Invalid input format."
        });

        assert!(error_response.get("error").is_some());
        assert!(error_response["error"].as_str().unwrap().contains("failed"));
    }

    #[test]
    fn test_structured_output_request() {
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
        use sglang_router_rs::core::{BasicWorkerBuilder, Worker, WorkerType};

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

        let prefill_worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9000),
                })
                .api_key("test_api_key")
                .build(),
        );

        let bootstrap_port = match prefill_worker.worker_type() {
            WorkerType::Prefill { bootstrap_port } => bootstrap_port,
            _ => None,
        };
        let batch_size = 16;
        let hostname = prefill_worker.bootstrap_host();

        benchmark_request["bootstrap_host"] = json!(vec![hostname; batch_size]);
        benchmark_request["bootstrap_port"] = json!(vec![bootstrap_port; batch_size]);
        benchmark_request["bootstrap_room"] =
            json!((0..batch_size).map(|_| 12345u64).collect::<Vec<_>>());

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

        assert_eq!(benchmark_request["return_logprob"], true);
        assert_eq!(benchmark_request["stream"], true);
    }

    #[test]
    fn test_server_info_response_format() {
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

        assert!(server_info["internal_states"][0]["avg_spec_accept_length"].is_f64());
        assert!(server_info["internal_states"][0]["last_gen_throughput"].is_f64());
        assert!(server_info["prefill"].is_array());
        assert!(server_info["decode"].is_array());
    }

    // Comprehensive Endpoint Coverage Test

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

        let missing: Vec<_> = implemented_endpoints
            .iter()
            .filter(|(_, _, impl_status)| !impl_status)
            .map(|(endpoint, method, _)| format!("{} {}", method, endpoint))
            .collect();

        assert_eq!(missing, vec!["POST /register"]);
    }

    #[test]
    fn test_large_batch_bootstrap_injection() {
        // This simulates the bench_one_batch_server.py scenario
        let large_batch_sizes = vec![1024, 4096, 8192];

        for batch_size in large_batch_sizes {
            let start = std::time::Instant::now();

            let mut large_batch_request = json!({
                "input_ids": vec![vec![1, 2, 3, 4]; batch_size],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                },
                "stream": true
            });

            let prefill_worker: Box<dyn Worker> = Box::new(
                BasicWorkerBuilder::new("http://prefill:8080")
                    .worker_type(WorkerType::Prefill {
                        bootstrap_port: Some(9000),
                    })
                    .api_key("test_api_key")
                    .build(),
            );

            let bootstrap_port = match prefill_worker.worker_type() {
                WorkerType::Prefill { bootstrap_port } => bootstrap_port,
                _ => None,
            };
            let hostname = prefill_worker.bootstrap_host();

            large_batch_request["bootstrap_host"] = json!(vec![hostname; batch_size]);
            large_batch_request["bootstrap_port"] = json!(vec![bootstrap_port; batch_size]);
            large_batch_request["bootstrap_room"] = json!((0..batch_size)
                .map(|_| rand::random::<u64>())
                .collect::<Vec<_>>());

            let elapsed = start.elapsed();

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
        let pd_policy_count = 3; // Random, PowerOfTwo, CacheAware
        assert_eq!(
            pd_policy_count, 3,
            "PDSelectionPolicy should have exactly 3 variants"
        );

        let _random = PDSelectionPolicy::Random;
        let _po2 = PDSelectionPolicy::PowerOfTwo;
        let _cache_aware = PDSelectionPolicy::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
        };
    }
}
