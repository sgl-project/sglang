use super::*;
use crate::common::mock_worker::MockWorker;
use http_body_util::BodyExt;

fn pd_ctx(prefill: &str, decode: &str, reorg: bool) -> Arc<AppContext> {
    let spec = |id: &str, url: &str, mode| WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: Some(8997),
    };
    let mut ctx = build_ctx(vec![
        spec("p", prefill, WorkerMode::Prefill),
        spec("d", decode, WorkerMode::Decode),
    ]);
    if reorg {
        let mutable = Arc::get_mut(&mut ctx).unwrap();
        mutable.config.model.policy = PolicyKind::PowerOfTwo;
        crate::common::use_reorg_factory(mutable);
    }
    ctx
}

fn chat(stream: bool) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "tiny", "stream": stream}).to_string(),
        ))
        .unwrap()
}

#[tokio::test]
async fn prefill_failure_stops_decode_without_waiting_for_bootstrap_timeout() {
    let decode = MockWorker::start_hanging(Duration::from_secs(10)).await;
    let rejected = json!({"error": "prefill rejected"});
    for reorg in [false, true] {
        for stream in [false, true] {
            let cases = [
                (
                    MockWorker::start_returning_error(StatusCode::BAD_REQUEST, rejected.clone())
                        .await,
                    StatusCode::BAD_REQUEST,
                ),
                (
                    MockWorker::start_returning_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        rejected.clone(),
                    )
                    .await,
                    StatusCode::BAD_GATEWAY,
                ),
                // Transport failure: the prefill body is cut off.
                (
                    MockWorker::start_returning_partial_body(StatusCode::OK, b"{").await,
                    StatusCode::BAD_GATEWAY,
                ),
            ];
            for (prefill, expected) in cases {
                let ctx = pd_ctx(&prefill.url, &decode.url, reorg);
                let response = tokio::time::timeout(
                    Duration::from_secs(1),
                    build_router(ctx.clone()).oneshot(chat(stream)),
                )
                .await
                .unwrap()
                .unwrap();
                assert_eq!(response.status(), expected);
                if expected == StatusCode::BAD_GATEWAY {
                    assert_eq!(response.headers()["x-router-error-code"], "prefill_failed");
                }
                let body = response.into_body().collect().await.unwrap().to_bytes();
                if expected == StatusCode::BAD_REQUEST {
                    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), rejected);
                }
                // The failure is charged to prefill, not to the cancelled decode.
                assert!(ctx.metrics.render().contains(&format!(
                    r#"worker_url="{}",model_id="tiny",mode="prefill""#,
                    prefill.url
                )));
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
        let ctx = pd_ctx(&prefill_url, &decode.url, true);
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            build_router(ctx.clone()).oneshot(chat(early_headers)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let sent = parse_body(decode.captured.lock().unwrap().last_body.as_ref().unwrap());
        tokio::time::timeout(Duration::from_secs(2), async {
            while decode.abort_log.lock().unwrap().is_empty()
                || ctx
                    .registry
                    .workers_for(&ModelId("tiny".into()))
                    .iter()
                    .any(|w| w.router_inflight_load() != 0)
            {
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
