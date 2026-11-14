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

    /// Prefill gets both requests at once; the second decode waits for a slot.
    #[tokio::test]
    async fn test_pd_decode_dispatch_limit_holds_decode_not_prefill() {
        use std::time::{Duration, Instant};

        use crate::common::mock_worker::take_request_arrivals;

        let (prefill_port, decode_port) = (19858, 19859);
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{prefill_port}"), None)],
                vec![format!("http://127.0.0.1:{decode_port}")],
            )
            .round_robin_policy()
            .decode_dispatch_limit(1)
            .host("127.0.0.1")
            .port(3858)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        let slow = |mut worker: MockWorkerConfig| {
            worker.response_delay_ms = 600;
            worker
        };
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                slow(TestWorkerConfig::prefill(prefill_port)),
                slow(TestWorkerConfig::decode(decode_port)),
            ],
        )
        .await;
        let app = ctx.create_app().await;
        take_request_arrivals(prefill_port);
        take_request_arrivals(decode_port);

        let send = || {
            let app = app.clone();
            async move {
                let request = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"text": "hi", "stream": false}).to_string(),
                    ))
                    .unwrap();
                app.oneshot(request).await.unwrap().status()
            }
        };
        let start = Instant::now();
        let statuses = tokio::join!(send(), send());
        assert_eq!(statuses, (StatusCode::OK, StatusCode::OK));

        let since_start = |port| {
            let mut arrivals: Vec<_> = take_request_arrivals(port)
                .into_iter()
                .map(|t| t - start)
                .collect();
            arrivals.sort();
            arrivals
        };
        let prefill = since_start(prefill_port);
        let decode = since_start(decode_port);
        assert!(
            prefill.len() == 2 && prefill[1] < Duration::from_millis(300),
            "{prefill:?}"
        );
        assert!(
            decode.len() == 2
                && decode[0] < Duration::from_millis(300)
                && decode[1] >= Duration::from_millis(500),
            "{decode:?}"
        );
        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod pd_responses_routing_tests {
    use std::{sync::Arc, time::Duration};

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
    use tokio::{
        io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
        net::{TcpListener, TcpStream},
        sync::{oneshot, Mutex},
        task::JoinHandle,
        time::timeout,
    };

    use super::*;

    const STORE_ERROR: &str = r#"{"error":{"message":"Response store is disabled. Stateful Responses require --enable-response-store on a standalone server; response storage is unavailable in PD mode.","type":"invalid_request_error","param":"previous_response_id","code":400}}"#;
    const DECODE_RESPONSE: &str =
        r#"{"id":"resp_decode","object":"response","status":"completed"}"#;
    const DECODE_STREAM: &str = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_decode\",\"status\":\"completed\"}}\n\n";

    #[derive(Clone, Copy)]
    enum WorkerReply {
        Json(StatusCode, &'static str),
        Stream(&'static str),
    }

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
            decode_gate: None,
            request_timeout: Duration::from_secs(600),
        }
    }

    fn register_workers(router: &PDRouter, prefill_url: String, decode_url: String) {
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
    }

    /// Spawn a local worker that records every JSON body POSTed to
    /// `/v1/responses` and returns either a JSON or an SSE response.
    async fn spawn_capture_worker(
        captured: Arc<Mutex<Vec<serde_json::Value>>>,
        reply: WorkerReply,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/v1/responses",
            post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let captured = captured.clone();
                async move {
                    // SRT forwards Responses.stream to ChatCompletionRequest's non-nullable bool.
                    let stream = body.get("stream").cloned().unwrap_or(json!(false));
                    if serde_json::from_value::<bool>(stream).is_err() {
                        return axum::response::Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .header(CONTENT_TYPE, "application/json")
                            .body(Body::from(
                                r#"{"error":{"message":"stream must be a boolean"}}"#,
                            ))
                            .unwrap();
                    }
                    captured.lock().await.push(body);
                    let (status, content_type, body) = match reply {
                        WorkerReply::Json(status, body) => (status, "application/json", body),
                        WorkerReply::Stream(body) => (StatusCode::OK, "text/event-stream", body),
                    };
                    axum::response::Response::builder()
                        .status(status)
                        .header(CONTENT_TYPE, content_type)
                        .header("content-length", body.len())
                        .body(Body::from(body))
                        .unwrap()
                }
            }),
        );
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), task)
    }

    fn responses_request(payload: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(payload).expect("valid responses request")
    }

    /// PDRouter must inject the PD bootstrap metadata into both worker
    /// requests, forward `disagg_prefill_dp_rank` to decode for a DP-aware
    /// prefill worker, target `/v1/responses` on both workers, and return the
    /// decode response.
    #[tokio::test]
    async fn test_pd_responses_default_buffered_request_injects_bootstrap_metadata() {
        let prefill_bodies = Arc::new(Mutex::new(Vec::new()));
        let decode_bodies = Arc::new(Mutex::new(Vec::new()));
        let (prefill_url, prefill_task) = spawn_capture_worker(
            prefill_bodies.clone(),
            WorkerReply::Json(StatusCode::OK, r#"{"id":"resp_prefill"}"#),
        )
        .await;
        let (decode_url, decode_task) = spawn_capture_worker(
            decode_bodies.clone(),
            WorkerReply::Json(StatusCode::OK, DECODE_RESPONSE),
        )
        .await;
        let router = make_pd_router();
        register_workers(&router, prefill_url, decode_url);
        let request = responses_request(json!({
            "model": "mock-model", "input": "Hello PD responses"
        }));
        assert_eq!(request.stream, None);

        let body = timeout(Duration::from_secs(5), async {
            let response = router.route_responses(None, &request, None).await;
            assert_eq!(response.status(), StatusCode::OK);
            response.into_body().collect().await.unwrap().to_bytes()
        })
        .await
        .expect("PD response must complete");
        assert_eq!(body.as_ref(), DECODE_RESPONSE.as_bytes());

        let prefill_bodies = prefill_bodies.lock().await;
        let decode_bodies = decode_bodies.lock().await;
        assert_eq!(prefill_bodies.len(), 1);
        assert_eq!(decode_bodies.len(), 1);
        let prefill_body = &prefill_bodies[0];
        let decode_body = &decode_bodies[0];
        for body in [prefill_body, decode_body] {
            assert_eq!(body["stream"], false);
            assert_eq!(body["input"], "Hello PD responses");
            assert_eq!(body["model"], "mock-model");
            assert_eq!(body["bootstrap_host"], "127.0.0.1");
            assert_eq!(body["bootstrap_port"], 8998);
            assert!(body["bootstrap_room"].is_u64());
        }
        assert_eq!(
            decode_body["bootstrap_room"],
            prefill_body["bootstrap_room"]
        );
        assert_eq!(prefill_body["data_parallel_rank"], 2);
        assert!(prefill_body.get("disagg_prefill_dp_rank").is_none());
        assert_eq!(decode_body["disagg_prefill_dp_rank"], 2);
        prefill_task.abort();
        decode_task.abort();
    }

    /// Streaming Responses requests must flow through the PD dual-dispatch
    /// path and return the decode SSE stream unchanged, with bootstrap
    /// metadata still injected into both worker requests.
    #[tokio::test]
    async fn test_pd_responses_streaming_passthrough() {
        let prefill_bodies = Arc::new(Mutex::new(Vec::new()));
        let decode_bodies = Arc::new(Mutex::new(Vec::new()));
        let (prefill_url, prefill_task) = spawn_capture_worker(
            prefill_bodies.clone(),
            WorkerReply::Stream("data: {\"id\":\"resp_prefill\"}\n\n"),
        )
        .await;
        let (decode_url, decode_task) =
            spawn_capture_worker(decode_bodies.clone(), WorkerReply::Stream(DECODE_STREAM)).await;
        let router = make_pd_router();
        register_workers(&router, prefill_url, decode_url);
        let request = responses_request(json!({
            "model": "mock-model", "input": "Hello PD streaming", "stream": true
        }));

        let body = timeout(Duration::from_secs(5), async {
            let response = router.route_responses(None, &request, None).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
            response.into_body().collect().await.unwrap().to_bytes()
        })
        .await
        .expect("PD stream must complete without a [DONE] sentinel");
        assert_eq!(body.as_ref(), DECODE_STREAM.as_bytes());
        let prefill_bodies = prefill_bodies.lock().await;
        let decode_bodies = decode_bodies.lock().await;
        assert_eq!(prefill_bodies.len(), 1);
        assert_eq!(decode_bodies.len(), 1);
        assert_eq!(decode_bodies[0]["stream"], true);
        assert!(prefill_bodies[0]["bootstrap_room"].is_u64());
        assert_eq!(
            decode_bodies[0]["bootstrap_room"],
            prefill_bodies[0]["bootstrap_room"]
        );
        prefill_task.abort();
        decode_task.abort();
    }

    #[tokio::test]
    async fn test_pd_responses_decode_error_is_valid_sse() {
        let (prefill_url, prefill_task) = spawn_capture_worker(
            Arc::new(Mutex::new(Vec::new())),
            WorkerReply::Json(StatusCode::OK, "{}"),
        )
        .await;
        let (decode_url, decode_task) = spawn_capture_worker(
            Arc::new(Mutex::new(Vec::new())),
            WorkerReply::Json(StatusCode::BAD_REQUEST, STORE_ERROR),
        )
        .await;
        let router = make_pd_router();
        register_workers(&router, prefill_url, decode_url);
        let request = responses_request(json!({
            "model":"mock-model", "input":"hi", "stream":true
        }));
        let body = timeout(Duration::from_secs(5), async {
            let response = router.route_responses(None, &request, None).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
            assert!(!response.headers().contains_key("content-length"));
            response.into_body().collect().await.unwrap().to_bytes()
        })
        .await
        .expect("decode error must complete");
        let body = std::str::from_utf8(&body).unwrap();
        let data = body
            .strip_suffix("\n\n")
            .expect("SSE event delimiter")
            .strip_prefix("data: ")
            .unwrap();
        let data: serde_json::Value = serde_json::from_str(data).unwrap();
        assert_eq!(data["error"]["status"], 400);
        assert_eq!(
            data["error"]["message"],
            serde_json::from_str::<serde_json::Value>(STORE_ERROR).unwrap()
        );
        prefill_task.abort();
        decode_task.abort();
    }

    /// Read the gateway's fixed-length JSON POST without an HTTP server that
    /// could hide disconnects by keeping a pending handler alive.
    async fn read_responses_post(socket: &mut BufReader<TcpStream>) -> serde_json::Value {
        let mut line = String::new();
        socket.read_line(&mut line).await.unwrap();
        assert_eq!(line, "POST /v1/responses HTTP/1.1\r\n");
        let mut content_length = None;
        loop {
            line.clear();
            assert_ne!(socket.read_line(&mut line).await.unwrap(), 0);
            if line == "\r\n" {
                break;
            }
            let (name, value) = line.split_once(':').unwrap();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = Some(value.trim().parse::<usize>().unwrap());
            }
        }
        let mut body = vec![0; content_length.expect("JSON POST must have Content-Length")];
        socket.read_exact(&mut body).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// Model #39122's pre-generation admission error without depending on its
    /// serving implementation. Prefill waits until decode has received the
    /// request, then rejects it without producing KV. Decode sends no headers
    /// or KV-timeout response: only a router disconnect can finish its task.
    #[tokio::test]
    async fn test_pd_responses_stateful_400_cancels_pending_decode() {
        for (param, value, stream) in [
            ("previous_response_id", json!("resp_previous"), false),
            ("previous_response_id", json!("resp_previous"), true),
            ("background", json!(true), true),
        ] {
            let prefill_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let decode_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let prefill_url = format!("http://{}", prefill_listener.local_addr().unwrap());
            let decode_url = format!("http://{}", decode_listener.local_addr().unwrap());
            let (decode_started_tx, decode_started_rx) = oneshot::channel();
            let admission_error = json!({
                "error": {
                    "message": "Response store is disabled. Stateful Responses require --enable-response-store on a standalone server; response storage is unavailable in PD mode.",
                    "type": "BadRequestError",
                    "param": param,
                    "code": 400
                }
            });
            let error_body = admission_error.to_string();
            let mut prefill_task = tokio::spawn(async move {
                let (socket, _) = prefill_listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                let body = read_responses_post(&mut socket).await;
                decode_started_rx.await.unwrap();
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    error_body.len(), error_body
                );
                socket
                    .get_mut()
                    .write_all(response.as_bytes())
                    .await
                    .unwrap();
                body
            });
            let mut decode_task = tokio::spawn(async move {
                let (socket, _) = decode_listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                let body = read_responses_post(&mut socket).await;
                decode_started_tx.send(()).unwrap();
                let mut byte = [0];
                assert_eq!(
                    socket.read(&mut byte).await.unwrap(),
                    0,
                    "router must close the pending decode connection after prefill rejects"
                );
                body
            });

            let router = make_pd_router();
            let prefill = Arc::new(
                BasicWorkerBuilder::new(prefill_url)
                    .worker_type(CoreWorkerType::Prefill {
                        bootstrap_port: Some(9001),
                    })
                    .build(),
            );
            let decode = Arc::new(
                BasicWorkerBuilder::new(decode_url)
                    .worker_type(CoreWorkerType::Decode)
                    .build(),
            );
            prefill.set_healthy(true);
            decode.set_healthy(true);
            router.worker_registry.register(prefill.clone());
            router.worker_registry.register(decode.clone());
            let mut payload = json!({
                "model": "mock-model",
                "input": "Continue the conversation",
                "stream": stream
            });
            payload[param] = value.clone();
            let request = responses_request(payload);

            // The timeout is only a regression guard. No worker timer, client
            // timeout, test teardown, or router drop can release decode here.
            let result = timeout(Duration::from_secs(5), async {
                let response = router.route_responses(None, &request, None).await;
                let status = response.status();
                let error_code = response.headers().get("x-smg-error-code").cloned();
                let body = response.into_body().collect().await.unwrap().to_bytes();
                let (prefill_body, decode_body) = tokio::join!(&mut prefill_task, &mut decode_task);
                (
                    status,
                    error_code,
                    body,
                    prefill_body.unwrap(),
                    decode_body.unwrap(),
                )
            })
            .await;
            if result.is_err() {
                // join! may already have consumed a completed handle. Only
                // abort and await tasks that are still pending on failure.
                for task in [&mut prefill_task, &mut decode_task] {
                    if !task.is_finished() {
                        task.abort();
                        let _ = task.await;
                    }
                }
            }
            let (status, error_code, body, prefill_body, decode_body) = result.expect(
                "prefill admission error must finish routing and disconnect decode promptly",
            );
            assert_eq!(status, StatusCode::BAD_REQUEST, "{param}, stream={stream}");
            assert_eq!(error_code.unwrap(), "prefill_bad_request");
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert!(body["error"]["message"]
                .as_str()
                .unwrap()
                .contains(&admission_error.to_string()));
            assert_eq!(prefill_body[param], value);
            assert_eq!(decode_body[param], value);
            assert!(prefill_body["bootstrap_room"].is_u64());
            assert_eq!(
                prefill_body["bootstrap_room"],
                decode_body["bootstrap_room"]
            );
            assert_eq!(prefill.load(), 0);
            assert_eq!(decode.load(), 0);
        }
    }

    /// Decode can reject during bootstrap before prefill produces KV. The
    /// pending prefill socket must close without any worker-side timeout.
    #[tokio::test]
    async fn test_pd_early_decode_failure_cancels_pending_prefill() {
        for stream in [false, true] {
            // None models a transport failure before response headers.
            for (status, sse_body, expected_status) in [
                (Some(400), None, 400),
                (Some(500), None, 500),
                (None, None, 502),
                (Some(200), Some("data: {\"text\":\"\",\"meta_info\":{\"finish_reason\":{\"type\":\"abort\",\"status_code\":500,\"message\":\"incompatible PD state\"}}}\n\n"), 500),
                (Some(200), Some("data: {\"error\":{\"message\":\"incompatible PD state\",\"code\":400}}\r\n\r\n"), 400),
            ] {
                if sse_body.is_some() && !stream { continue; }
                let prefill_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let decode_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let prefill_url = format!("http://{}", prefill_listener.local_addr().unwrap());
                let decode_url = format!("http://{}", decode_listener.local_addr().unwrap());
                let (prefill_started_tx, prefill_started_rx) = oneshot::channel();
                let mut prefill_task = tokio::spawn(async move {
                    let (socket, _) = prefill_listener.accept().await.unwrap();
                    let mut socket = BufReader::new(socket);
                    read_responses_post(&mut socket).await;
                    prefill_started_tx.send(()).unwrap();
                    let mut byte = [0];
                    assert_eq!(socket.read(&mut byte).await.unwrap(), 0);
                });
                let decode_task = tokio::spawn(async move {
                    let (socket, _) = decode_listener.accept().await.unwrap();
                    let mut socket = BufReader::new(socket);
                    read_responses_post(&mut socket).await;
                    prefill_started_rx.await.unwrap();
                    if let Some(status) = status {
                        if let Some(sse_body) = sse_body {
                            socket.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
                            let payload = format!(": heartbeat\r\n\r\n{sse_body}");
                            for piece in payload.as_bytes().chunks(17) {
                                socket.get_mut().write_all(format!("{:x}\r\n", piece.len()).as_bytes()).await.unwrap();
                                socket.get_mut().write_all(piece).await.unwrap();
                                socket.get_mut().write_all(b"\r\n").await.unwrap();
                                tokio::task::yield_now().await;
                            }
                            // No EOF/terminating chunk: first-event rejection
                            // must cancel the still-open decode stream too.
                            assert_eq!(socket.read(&mut [0]).await.unwrap(), 0);
                            return;
                        }
                        let body = r#"{"error":"incompatible PD state"}"#;
                        let response = format!(
                            "HTTP/1.1 {status} Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        socket
                            .get_mut()
                            .write_all(response.as_bytes())
                            .await
                            .unwrap();
                    }
                });
                let mut router = make_pd_router();
                router.retry_config.max_retries = 0;
                let prefill = Arc::new(
                    BasicWorkerBuilder::new(prefill_url)
                        .worker_type(CoreWorkerType::Prefill {
                            bootstrap_port: Some(9001),
                        })
                        .build(),
                );
                let decode = Arc::new(
                    BasicWorkerBuilder::new(decode_url)
                        .worker_type(CoreWorkerType::Decode)
                        .build(),
                );
                prefill.set_healthy(true);
                decode.set_healthy(true);
                router.worker_registry.register(prefill.clone());
                router.worker_registry.register(decode.clone());
                let request = responses_request(json!({
                    "model":"mock-model", "input":"hi", "stream":stream
                }));
                let result = timeout(Duration::from_secs(5), async {
                    let response = router.route_responses(None, &request, None).await;
                    assert_eq!(response.status().as_u16(), expected_status);
                    let body = response.into_body().collect().await.unwrap().to_bytes();
                    if status.is_some() {
                        assert!(std::str::from_utf8(&body)
                            .unwrap()
                            .contains("incompatible PD state"));
                    }
                    (&mut prefill_task).await.unwrap();
                })
                .await;
                if result.is_err() {
                    prefill_task.abort();
                    decode_task.abort();
                }
                result.expect("decode failure must return and cancel prefill promptly");
                decode_task.await.unwrap();
                assert_eq!(prefill.load(), 0);
                assert_eq!(decode.load(), 0);
                let prefill_stats = prefill.circuit_breaker().stats();
                assert_eq!(prefill_stats.total_failures, 0);
                assert_eq!(prefill_stats.total_successes, 0);
                let decode_stats = decode.circuit_breaker().stats();
                assert_eq!(decode_stats.total_failures, u64::from(expected_status >= 500));
                assert_eq!(decode_stats.total_successes, u64::from(expected_status < 500));
            }
        }
    }

    #[tokio::test]
    async fn test_pd_early_decode_success_still_waits_for_prefill() {
        for stream in [false, true] {
            let prefill_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let decode_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let prefill_url = format!("http://{}", prefill_listener.local_addr().unwrap());
            let decode_url = format!("http://{}", decode_listener.local_addr().unwrap());
            let (release_prefill_tx, release_prefill_rx) = oneshot::channel();
            let (decode_done_tx, decode_done_rx) = oneshot::channel();
            let prefill_task = tokio::spawn(async move {
                let (socket, _) = prefill_listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                read_responses_post(&mut socket).await;
                release_prefill_rx.await.unwrap();
                socket.get_mut().write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                ).await.unwrap();
            });
            let expected = if stream {
                "data: {\"error\":null,\"type\":\"response.completed\"}\n\n"
            } else {
                DECODE_RESPONSE
            };
            let decode_task = tokio::spawn(async move {
                let (socket, _) = decode_listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                read_responses_post(&mut socket).await;
                let content_type = if stream {
                    "text/event-stream"
                } else {
                    "application/json"
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{expected}",
                    expected.len()
                );
                socket
                    .get_mut()
                    .write_all(response.as_bytes())
                    .await
                    .unwrap();
                decode_done_tx.send(()).unwrap();
            });
            let router = make_pd_router();
            register_workers(&router, prefill_url, decode_url);
            let request = responses_request(json!({
                "model":"mock-model", "input":"hi", "stream":stream
            }));
            let response = router.route_responses(None, &request, None);
            tokio::pin!(response);
            tokio::select! {
                _ = &mut response => panic!("successful decode must wait for prefill"),
                _ = decode_done_rx => {}
                _ = tokio::time::sleep(Duration::from_secs(5)) => panic!("decode request must reach worker"),
            }
            assert!(timeout(Duration::from_millis(50), &mut response)
                .await
                .is_err());
            release_prefill_tx.send(()).unwrap();
            let body = timeout(Duration::from_secs(5), async {
                let response = response.await;
                assert_eq!(response.status(), StatusCode::OK);
                response.into_body().collect().await.unwrap().to_bytes()
            })
            .await
            .expect("successful PD response must finish after prefill");
            assert_eq!(body.as_ref(), expected.as_bytes());
            prefill_task.await.unwrap();
            decode_task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_pd_deferred_decode_selection_failure_does_not_penalise_prefill() {
        use smg::routers::http::decode_gate::DecodeGate;

        let (prefill_url, prefill_task) = spawn_capture_worker(
            Arc::new(Mutex::new(Vec::new())),
            WorkerReply::Json(StatusCode::OK, "{}"),
        )
        .await;
        let mut router = make_pd_router();
        router.retry_config.max_retries = 0;
        router.decode_gate = Some(Arc::new(DecodeGate::new(1)));
        let prefill = Arc::new(
            BasicWorkerBuilder::new(prefill_url)
                .worker_type(CoreWorkerType::Prefill {
                    bootstrap_port: Some(9001),
                })
                .build(),
        );
        prefill.set_healthy(true);
        router.worker_registry.register(prefill.clone());
        let request = responses_request(json!({"model":"mock-model", "input":"hi"}));
        let response = timeout(
            Duration::from_secs(5),
            router.route_responses(None, &request, None),
        )
        .await
        .expect("missing decode worker must fail promptly");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(prefill.load(), 0);
        assert_eq!(prefill.circuit_breaker().stats().total_failures, 0);
        assert_eq!(prefill.circuit_breaker().stats().total_successes, 0);
        prefill_task.abort();
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
