use super::*;
use crate::common::mock_worker::MockWorker;
use http_body_util::BodyExt;

fn spec(id: &str, url: &str, mode: WorkerMode, group: Option<&str>) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: Some(8997),
        transfer_group: group.map(str::to_owned),
    }
}

#[tokio::test]
async fn prefill_failure_stops_decode_without_waiting_for_bootstrap_timeout() {
    let decode = MockWorker::start_hanging(Duration::from_secs(10)).await;
    for streaming in [false, true] {
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
        ] {
            let prefill =
                MockWorker::start_returning_error(status, json!({"error": "prefill rejected"}))
                    .await;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let closed_url = format!("http://{}", listener.local_addr().unwrap());
            drop(listener);
            let url = if status == StatusCode::BAD_GATEWAY {
                &closed_url
            } else {
                &prefill.url
            };
            let ctx = build_ctx(vec![
                spec("p", url, WorkerMode::Prefill, None),
                spec("d", &decode.url, WorkerMode::Decode, None),
            ]);
            let request = Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "tiny", "stream": streaming}).to_string(),
                ))
                .unwrap();
            let response = tokio::time::timeout(
                Duration::from_secs(1),
                build_router(ctx.clone()).oneshot(request),
            )
            .await
            .unwrap()
            .unwrap();
            let expected = if status.is_client_error() {
                status
            } else {
                StatusCode::BAD_GATEWAY
            };
            assert_eq!(response.status(), expected);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            if status.is_client_error() {
                assert_eq!(
                    serde_json::from_slice::<Value>(&body).unwrap(),
                    json!({"error": "prefill rejected"})
                );
            }
            for worker in ctx.registry.workers_for(&ModelId("tiny".into())) {
                assert_eq!(worker.router_inflight_load(), 0);
            }
        }
    }
}

#[tokio::test]
async fn transfer_groups_require_a_complete_pair_and_gate_readiness() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    for (p, d) in [
        (None, None),
        (Some("a"), Some("a")),
        (Some("a"), Some("b")),
        (None, Some("a")),
        (Some("a"), None),
    ] {
        let ctx = build_ctx(vec![
            spec("orphan", &prefill.url, WorkerMode::Prefill, Some("orphan")),
            spec("p", &prefill.url, WorkerMode::Prefill, p),
            spec("d", &decode.url, WorkerMode::Decode, d),
        ]);
        ctx.mark_ready();
        let expected = if p == d {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
        let app = build_router(ctx);
        assert_eq!(
            app.clone()
                .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status(),
            expected
        );
        assert_eq!(
            app.oneshot(chat_request()).await.unwrap().status(),
            expected
        );
    }
}
