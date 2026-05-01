//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use serde_json::{json, Value};
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
                    routed_experts_b64: None,
                    always_emit_routed_experts: false,
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

    /// Verify the end-to-end PD merge of `routed_experts` for chat
    /// completions. The prefill worker emits a base64 prefix; the decode
    /// worker emits the prefix plus per-token suffix bytes; the gateway
    /// must concatenate `prefill_bytes ++ decode_bytes[prefill_len..]`
    /// into the response payload under `sglext.routed_experts`.
    ///
    /// This test exercises the full chain
    /// `v1_chat_completions -> RouterTrait::route_chat -> PDRouter ->
    ///  execute_dual_dispatch -> merge_prefill_json` and would catch a
    /// regression where `body_raw` collapses to `None` upstream (every
    /// existing pd_router unit test would still pass).
    #[tokio::test]
    async fn test_pd_chat_completion_merges_routed_experts() {
        let prefill_bytes = vec![1u8, 2, 3, 4];
        let decode_bytes = vec![1u8, 2, 3, 4, 9, 9, 9, 9];
        let prefill_b64 = BASE64_STANDARD.encode(&prefill_bytes);
        let decode_b64 = BASE64_STANDARD.encode(&decode_bytes);

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19840".to_string(), None)],
                vec!["http://127.0.0.1:19841".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3840)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let mut prefill_worker = TestWorkerConfig::prefill(19840);
        prefill_worker.routed_experts_b64 = Some(prefill_b64.clone());
        let mut decode_worker = TestWorkerConfig::decode(19841);
        decode_worker.routed_experts_b64 = Some(decode_b64.clone());

        let ctx =
            AppTestContext::new_with_config(config, vec![prefill_worker, decode_worker]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "return_routed_experts": true,
            "stream": false,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "PD chat completion with return_routed_experts should succeed"
        );

        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value =
            serde_json::from_slice(&body_bytes).expect("response body should be JSON");

        let merged_b64 = body
            .pointer("/sglext/routed_experts")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("expected sglext.routed_experts in merged response: {body}"));
        let merged_bytes = BASE64_STANDARD
            .decode(merged_b64)
            .expect("merged routed_experts should be valid base64");

        // Expected: prefill prefix + decode-suffix beyond prefill_len.
        let mut expected = prefill_bytes.clone();
        expected.extend_from_slice(&decode_bytes[prefill_bytes.len()..]);
        assert_eq!(
            merged_bytes, expected,
            "merged routed_experts should be prefill prefix + decode suffix",
        );

        ctx.shutdown().await;
    }

    /// Same shape as `test_pd_chat_completion_merges_routed_experts` but
    /// for `/generate`. The merge helper inspects both `meta_info` and
    /// `sglext`; the chat test exercises the `sglext` branch and this
    /// test exercises the `meta_info` branch. A regression that broke
    /// just the `meta_info` path (or that forgot to thread `body_raw`
    /// into `route_generate`) would slip past the chat test.
    #[tokio::test]
    async fn test_pd_generate_merges_routed_experts() {
        let prefill_bytes = vec![1u8, 2, 3, 4];
        let decode_bytes = vec![1u8, 2, 3, 4, 9, 9, 9, 9];
        let prefill_b64 = BASE64_STANDARD.encode(&prefill_bytes);
        let decode_b64 = BASE64_STANDARD.encode(&decode_bytes);

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19842".to_string(), None)],
                vec!["http://127.0.0.1:19843".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3842)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let mut prefill_worker = TestWorkerConfig::prefill(19842);
        prefill_worker.routed_experts_b64 = Some(prefill_b64);
        let mut decode_worker = TestWorkerConfig::decode(19843);
        decode_worker.routed_experts_b64 = Some(decode_b64);

        let ctx =
            AppTestContext::new_with_config(config, vec![prefill_worker, decode_worker]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "hi",
            "return_routed_experts": true,
            "stream": false,
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
            "PD /generate with return_routed_experts should succeed"
        );

        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body_bytes).unwrap();

        let merged_b64 = body
            .pointer("/meta_info/routed_experts")
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                panic!("expected meta_info.routed_experts in merged response: {body}")
            });
        let merged_bytes = BASE64_STANDARD.decode(merged_b64).unwrap();

        let mut expected = prefill_bytes.clone();
        expected.extend_from_slice(&decode_bytes[prefill_bytes.len()..]);
        assert_eq!(merged_bytes, expected);

        ctx.shutdown().await;
    }

    /// `return_routed_experts` must be a bool. The PD router parses it
    /// into a strict-typed `SglangExtensions` and surfaces type errors as
    /// a 400 — protects against a regression where the parse error gets
    /// swallowed and the request silently runs without merge enabled.
    /// `route_chat` and `route_generate` each have their own copy of the
    /// parse-then-`bad_request` ladder, so this test covers chat and the
    /// next one covers `/generate`.
    #[tokio::test]
    async fn test_pd_chat_rejects_invalid_extension_type() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19844".to_string(), None)],
                vec!["http://127.0.0.1:19845".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3844)
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
                TestWorkerConfig::prefill(19844),
                TestWorkerConfig::decode(19845),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "return_routed_experts": "yes",
            "stream": false,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "non-bool return_routed_experts should yield 400"
        );

        ctx.shutdown().await;
    }

    /// Companion of the chat test: `route_generate` has its own copy of
    /// the parse-then-`bad_request` ladder. A regression that loosens
    /// only the generate-side parser (e.g. log-and-default-false) would
    /// slip past the chat test.
    #[tokio::test]
    async fn test_pd_generate_rejects_invalid_extension_type() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19850".to_string(), None)],
                vec!["http://127.0.0.1:19851".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3850)
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
                TestWorkerConfig::prefill(19850),
                TestWorkerConfig::decode(19851),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "hi",
            "return_routed_experts": "yes",
            "stream": false,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }

    /// Streaming PD with `return_routed_experts: true` is unsupported:
    /// the SSE pipeline does not merge routed_experts across
    /// prefill/decode, so silently honouring the flag would return
    /// decode-only bytes with no signal to the user. The router rejects
    /// with 400 up front. This test covers the chat-completions branch.
    #[tokio::test]
    async fn test_pd_chat_streaming_rejects_routed_experts() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19852".to_string(), None)],
                vec!["http://127.0.0.1:19853".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3852)
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
                TestWorkerConfig::prefill(19852),
                TestWorkerConfig::decode(19853),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "return_routed_experts": true,
            "stream": true,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "streaming + return_routed_experts should yield 400"
        );

        ctx.shutdown().await;
    }

    /// Companion of the chat streaming reject test, covering `/generate`.
    /// `route_generate` has its own copy of the streaming-incompat guard.
    #[tokio::test]
    async fn test_pd_generate_streaming_rejects_routed_experts() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19854".to_string(), None)],
                vec!["http://127.0.0.1:19855".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3854)
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
                TestWorkerConfig::prefill(19854),
                TestWorkerConfig::decode(19855),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "hi",
            "return_routed_experts": true,
            "stream": true,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }

    /// Inverse of the merge tests: when `return_routed_experts` is
    /// absent, the gateway must NOT merge — the response should mirror
    /// what the decode worker produced. Catches a regression that
    /// always-merges regardless of the flag.
    #[tokio::test]
    async fn test_pd_does_not_merge_when_flag_absent() {
        // Asymmetric so a stray merge would be visible: prefill bytes
        // are not a prefix of decode bytes, so concatenation would
        // produce something different from either side alone.
        let prefill_b64 = BASE64_STANDARD.encode(b"PPPP");
        let decode_b64 = BASE64_STANDARD.encode(b"DDDD");

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19846".to_string(), None)],
                vec!["http://127.0.0.1:19847".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3846)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        // always_emit_routed_experts bypasses the request-flag gate so
        // both workers emit experts regardless of what the request asked
        // for. That's how this test observes "no merge happened" — the
        // response must equal decode's bytes alone.
        let mut prefill_worker = TestWorkerConfig::prefill(19846);
        prefill_worker.routed_experts_b64 = Some(prefill_b64);
        prefill_worker.always_emit_routed_experts = true;
        let mut decode_worker = TestWorkerConfig::decode(19847);
        decode_worker.routed_experts_b64 = Some(decode_b64.clone());
        decode_worker.always_emit_routed_experts = true;

        let ctx =
            AppTestContext::new_with_config(config, vec![prefill_worker, decode_worker]).await;
        let app = ctx.create_app().await;

        // Note: no return_routed_experts field at all.
        let payload = json!({
            "text": "hi",
            "stream": false,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body_bytes).unwrap();

        let response_b64 = body
            .pointer("/meta_info/routed_experts")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("expected meta_info.routed_experts in response: {body}"));
        assert_eq!(
            response_b64, decode_b64,
            "without the flag, response should pass through decode's routed_experts unchanged"
        );

        ctx.shutdown().await;
    }

    /// SGLang fields the typed openai-protocol struct drops — like
    /// `routed_dp_rank`, `disagg_prefill_dp_rank`, or any future
    /// extension — must reach the backend verbatim. The PD router does
    /// this by forwarding the raw request bytes; this test pins that
    /// contract by inspecting what the mock worker actually received.
    #[tokio::test]
    async fn test_pd_forwards_unknown_sglang_extensions() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19848".to_string(), None)],
                vec!["http://127.0.0.1:19849".to_string()],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3848)
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
                TestWorkerConfig::prefill(19848),
                TestWorkerConfig::decode(19849),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        // routed_dp_rank is in EXTENSION_FIELD_NAMES but NOT in typed
        // SglangExtensions, so it'll only survive if the gateway forwards
        // raw bytes. some_future_field tests the forward-compat case.
        let payload = json!({
            "text": "hi",
            "stream": false,
            "routed_dp_rank": 7,
            "some_future_field": {"nested": [1, 2, 3]},
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let prefill_recv = ctx.workers[0].recorded_requests();
        assert_eq!(
            prefill_recv.len(),
            1,
            "prefill worker should have received exactly one request"
        );
        let received = &prefill_recv[0];
        assert_eq!(
            received.get("routed_dp_rank"),
            Some(&json!(7)),
            "routed_dp_rank should pass through to backend (received: {received})"
        );
        assert_eq!(
            received.get("some_future_field"),
            Some(&json!({"nested": [1, 2, 3]})),
            "unknown fields should pass through to backend (received: {received})"
        );

        ctx.shutdown().await;
    }
}
