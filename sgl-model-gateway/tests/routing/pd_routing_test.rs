//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod pd_routing_tests {
    use super::*;

    /// Test basic PD mode routing with prefill and decode workers
    #[tokio::test]
    async fn test_pd_mode_basic_routing() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://127.0.0.1:19800".to_string(), None),
                    ("http://127.0.0.1:19801".to_string(), None),
                ],
                vec![
                    "http://127.0.0.1:19802".to_string(),
                    "http://127.0.0.1:19803".to_string(),
                ],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3800)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        // Note: For PD mode tests, we need to start prefill and decode workers separately
        // The test context will need to handle this specially
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill workers
                TestWorkerConfig::prefill(19800),
                TestWorkerConfig::prefill(19801),
                // Decode workers
                TestWorkerConfig::decode(19802),
                TestWorkerConfig::decode(19803),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        for i in 0..10 {
            let payload = json!({
                "text": format!("PD mode request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "PD mode request should succeed"
            );
        }

        ctx.shutdown().await;
    }

    /// Test PD mode with round robin policy
    #[tokio::test]
    async fn test_pd_mode_round_robin() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19810".to_string(), None)],
                vec![
                    "http://127.0.0.1:19811".to_string(),
                    "http://127.0.0.1:19812".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3801)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19810),
                TestWorkerConfig::decode(19811),
                TestWorkerConfig::decode(19812),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        for i in 0..20 {
            let payload = json!({
                "text": format!("PD round robin {}", i),
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

        assert_eq!(
            success_count, 20,
            "All requests should succeed in PD mode with round robin"
        );

        ctx.shutdown().await;
    }

    /// Test PD mode handles worker failures gracefully
    #[tokio::test]
    async fn test_pd_mode_with_failing_decode_worker() {
        use smg::config::RetryConfig;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19820".to_string(), None)],
                vec![
                    "http://127.0.0.1:19821".to_string(),
                    "http://127.0.0.1:19822".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3802)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19820),
                MockWorkerConfig {
                    port: 19821,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0, // Failing decode worker
                },
                TestWorkerConfig::decode(19822), // Healthy decode worker
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Request should succeed via retry to healthy decode worker
        let payload = json!({
            "text": "Test with failing decode worker",
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
            "Request should succeed via retry to healthy decode worker"
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod pd_responses_routing_tests {
    use std::sync::Arc;

    use axum::routing::post;
    use http_body_util::BodyExt;
    use smg::{
        config::{PolicyConfig, RetryConfig},
        core::{
            BasicWorkerBuilder, DPAwareWorkerBuilder, Worker, WorkerRegistry,
            WorkerType as CoreWorkerType,
        },
        policies::PolicyRegistry,
        protocols::responses::ResponsesRequest,
        routers::{pd_router::PDRouter, RouterTrait},
    };
    use tokio::{net::TcpListener, sync::Mutex};

    use super::*;

    /// Regression test: `/v1/responses` must be routed through the PD
    /// dual-dispatch path instead of falling back to the `RouterTrait`
    /// default `501 NOT_IMPLEMENTED` implementation.
    #[tokio::test]
    async fn test_pd_mode_responses_routing() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19830".to_string(), None)],
                vec!["http://127.0.0.1:19831".to_string()],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3803)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19830),
                TestWorkerConfig::decode(19831),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "input": "PD mode responses request",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "PD mode /v1/responses request should be dual-dispatched, not 501"
        );

        ctx.shutdown().await;
    }

    fn make_pd_router() -> PDRouter {
        PDRouter {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            client: reqwest::Client::new(),
            retry_config: RetryConfig::default(),
            api_key: None,
            enable_igw: false,
        }
    }

    /// Spawn a local worker that records every JSON body POSTed to
    /// `/v1/responses` and returns either a JSON or an SSE response.
    async fn spawn_capture_worker(
        captured: Arc<Mutex<Vec<serde_json::Value>>>,
        streaming: bool,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/v1/responses",
            post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let captured = captured.clone();
                async move {
                    captured.lock().await.push(body);
                    if streaming {
                        axum::response::Response::builder()
                            .status(StatusCode::OK)
                            .header(CONTENT_TYPE, "text/event-stream")
                            .body(Body::from(
                                "data: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n",
                            ))
                            .unwrap()
                    } else {
                        axum::response::Response::builder()
                            .status(StatusCode::OK)
                            .header(CONTENT_TYPE, "application/json")
                            .body(Body::from(
                                serde_json::to_string(&json!({
                                    "id": "resp_mock_decode",
                                    "object": "response",
                                    "status": "completed"
                                }))
                                .unwrap(),
                            ))
                            .unwrap()
                    }
                }
            }),
        );
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{}", addr)
    }

    fn responses_request(payload: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(payload).expect("valid responses request")
    }

    /// PDRouter must inject the PD bootstrap metadata into both worker
    /// requests, forward `disagg_prefill_dp_rank` to decode for a DP-aware
    /// prefill worker, target `/v1/responses` on both workers, and return the
    /// decode response.
    #[tokio::test]
    async fn test_pd_responses_injects_bootstrap_metadata() {
        let prefill_bodies = Arc::new(Mutex::new(Vec::new()));
        let decode_bodies = Arc::new(Mutex::new(Vec::new()));
        let prefill_url = spawn_capture_worker(prefill_bodies.clone(), false).await;
        let decode_url = spawn_capture_worker(decode_bodies.clone(), false).await;

        let router = make_pd_router();
        let prefill = DPAwareWorkerBuilder::new(prefill_url, 2, 4)
            .worker_type(CoreWorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        prefill.set_healthy(true);
        let decode = BasicWorkerBuilder::new(decode_url)
            .worker_type(CoreWorkerType::Decode)
            .build();
        decode.set_healthy(true);
        router.worker_registry.register(Arc::new(prefill));
        router.worker_registry.register(Arc::new(decode));

        let request = responses_request(json!({
            "model": "mock-model",
            "input": "Hello PD responses",
            "stream": false
        }));

        let response = router.route_responses(None, &request, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body_json["id"], "resp_mock_decode",
            "decode response should be passed through"
        );

        let prefill_bodies = prefill_bodies.lock().await;
        let decode_bodies = decode_bodies.lock().await;
        assert_eq!(
            prefill_bodies.len(),
            1,
            "prefill should receive one request"
        );
        assert_eq!(decode_bodies.len(), 1, "decode should receive one request");
        let prefill_body = &prefill_bodies[0];
        let decode_body = &decode_bodies[0];

        // Original Responses fields must be preserved.
        assert_eq!(prefill_body["input"], "Hello PD responses");
        assert_eq!(prefill_body["model"], "mock-model");
        assert_eq!(decode_body["input"], "Hello PD responses");
        assert_eq!(decode_body["model"], "mock-model");

        // Bootstrap metadata must be injected for both workers, with the same
        // room so prefill and decode can rendezvous.
        assert_eq!(prefill_body["bootstrap_host"], "127.0.0.1");
        assert_eq!(prefill_body["bootstrap_port"], 8998);
        assert!(prefill_body["bootstrap_room"].is_u64());
        assert_eq!(decode_body["bootstrap_host"], "127.0.0.1");
        assert_eq!(decode_body["bootstrap_port"], 8998);
        assert_eq!(
            decode_body["bootstrap_room"],
            prefill_body["bootstrap_room"]
        );

        // DP-aware prefill: its own rank goes to the prefill request, and
        // decode learns which prefill DP worker holds the KV cache.
        assert_eq!(prefill_body["data_parallel_rank"], 2);
        assert!(prefill_body.get("disagg_prefill_dp_rank").is_none());
        assert_eq!(decode_body["disagg_prefill_dp_rank"], 2);
    }

    /// Streaming Responses requests must flow through the PD dual-dispatch
    /// path and return the decode SSE stream unchanged, with bootstrap
    /// metadata still injected into both worker requests. `background=true`
    /// with `stream=true` stays attached to the connection, so it must not
    /// be rejected by the router.
    #[tokio::test]
    async fn test_pd_responses_streaming_passthrough() {
        let prefill_bodies = Arc::new(Mutex::new(Vec::new()));
        let decode_bodies = Arc::new(Mutex::new(Vec::new()));
        let prefill_url = spawn_capture_worker(prefill_bodies.clone(), false).await;
        let decode_url = spawn_capture_worker(decode_bodies.clone(), true).await;

        let router = make_pd_router();
        let prefill = BasicWorkerBuilder::new(prefill_url)
            .worker_type(CoreWorkerType::Prefill {
                bootstrap_port: Some(9001),
            })
            .build();
        prefill.set_healthy(true);
        let decode = BasicWorkerBuilder::new(decode_url)
            .worker_type(CoreWorkerType::Decode)
            .build();
        decode.set_healthy(true);
        router.worker_registry.register(Arc::new(prefill));
        router.worker_registry.register(Arc::new(decode));

        let request = responses_request(json!({
            "model": "mock-model",
            "input": "Hello PD streaming",
            "stream": true,
            "background": true
        }));

        let response = router.route_responses(None, &request, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);
        assert!(
            body_text.contains("response.completed"),
            "decode SSE stream should be passed through, got: {body_text:?}"
        );

        let prefill_bodies = prefill_bodies.lock().await;
        let decode_bodies = decode_bodies.lock().await;
        assert_eq!(prefill_bodies.len(), 1);
        assert_eq!(decode_bodies.len(), 1);
        assert_eq!(decode_bodies[0]["stream"], true);
        assert_eq!(decode_bodies[0]["background"], true);
        assert!(prefill_bodies[0]["bootstrap_room"].is_u64());
        assert_eq!(
            decode_bodies[0]["bootstrap_room"],
            prefill_bodies[0]["bootstrap_room"]
        );
    }

    /// Detached background responses (`background=true, stream=false`) are
    /// retrieved through the /v1/responses/{id} endpoints, which the PD router
    /// does not implement. They must be rejected deterministically instead of
    /// dual-dispatched.
    #[tokio::test]
    async fn test_pd_responses_detached_background_rejected() {
        let router = make_pd_router();

        let request = responses_request(json!({
            "model": "mock-model",
            "input": "Hello background",
            "background": true,
            "stream": false
        }));

        let response = router.route_responses(None, &request, None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response
                .headers()
                .get("x-smg-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("pd_unsupported_background_responses")
        );
    }
}
