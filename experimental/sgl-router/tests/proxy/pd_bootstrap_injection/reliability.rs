use super::*;
use crate::common::mock_worker::MockWorker;
use http_body_util::BodyExt;

fn spec(id: &str, url: &str, mode: WorkerMode) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: Some(8997),
    }
}

#[tokio::test]
async fn prefill_failure_stops_decode_without_waiting_for_bootstrap_timeout() {
    let decode = MockWorker::start_hanging(Duration::from_secs(10)).await;
    for reorg in [true, false] {
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
                let mut ctx = build_ctx(vec![
                    spec("p", url, WorkerMode::Prefill),
                    spec("d", &decode.url, WorkerMode::Decode),
                ]);
                if reorg {
                    let mutable = Arc::get_mut(&mut ctx).unwrap();
                    mutable.config.model.policy = PolicyKind::PowerOfTwo;
                    let state = sgl_router::state::kv_events::KvEventIndex::new();
                    let (resolver, _) = sgl_router::policies_reorg::factory::build_resolver(
                        &mutable.config.model,
                        &state,
                        None,
                    )
                    .unwrap();
                    mutable.chat_routing = sgl_router::server::app_context::ChatRouting::Reorg(
                        [(ModelId("tiny".into()), resolver)].into(),
                    );
                }
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
}

/// Prefill waits until decode has started, then rejects the request. Verify
/// engine cleanup both before decode headers and after its SSE pump exists.
#[tokio::test]
async fn reorg_prefill_failure_aborts_started_decode_and_releases_load() {
    for early_headers in [false, true] {
        let decode = if early_headers {
            MockWorker::start_slow_stream(vec!["data: still waiting\n\n"], Duration::from_secs(10))
                .await
        } else {
            MockWorker::start_hanging(Duration::from_secs(10)).await
        };
        let captured = decode.captured.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let prefill_url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let app = axum::Router::new().route(
                "/v1/chat/completions",
                axum::routing::post(move || {
                    let captured = captured.clone();
                    async move {
                        while captured.lock().unwrap().last_body.is_none() {
                            tokio::task::yield_now().await;
                        }
                        // Give the streaming response time to reach the router before prefill fails.
                        if early_headers {
                            tokio::time::sleep(Duration::from_millis(30)).await;
                        }
                        (
                            StatusCode::BAD_REQUEST,
                            axum::Json(json!({"error":"prefill rejected"})),
                        )
                    }
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });
        let mut ctx = build_ctx(vec![
            spec("p", &prefill_url, WorkerMode::Prefill),
            spec("d", &decode.url, WorkerMode::Decode),
        ]);
        let mutable = Arc::get_mut(&mut ctx).unwrap();
        mutable.config.model.policy = PolicyKind::PowerOfTwo;
        let state = sgl_router::state::kv_events::KvEventIndex::new();
        let (resolver, _) = sgl_router::policies_reorg::factory::build_resolver(
            &mutable.config.model,
            &state,
            None,
        )
        .unwrap();
        mutable.chat_routing = sgl_router::server::app_context::ChatRouting::Reorg(
            [(ModelId("tiny".into()), resolver)].into(),
        );
        let request = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"tiny", "stream":early_headers}).to_string(),
            ))
            .unwrap();
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            build_router(ctx.clone()).oneshot(request),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let sent = parse_body(decode.captured.lock().unwrap().last_body.as_ref().unwrap());
        assert!(sent["rid"].is_string());
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if !decode.abort_log.lock().unwrap().is_empty()
                    && ctx
                        .registry
                        .workers_for(&ModelId("tiny".into()))
                        .iter()
                        .all(|w| w.router_inflight_load() == 0)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            decode.abort_log.lock().unwrap()[0],
            json!({"rid":sent["rid"], "abort_all":false})
        );
        server.abort();
    }
}
