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

/// Either side's failure ends the request without waiting for the other side.
#[tokio::test]
async fn pd_failure_returns_without_waiting_for_the_other_side() {
    let rejected = json!({"error": "rejected"});
    let error = |status| MockWorker::start_returning_error(status, rejected.clone());
    let hanging = || MockWorker::start_hanging(Duration::from_secs(10));
    for reorg in [false, true] {
        for stream in [false, true] {
            // (prefill, decode, client status, whether prefill is blamed)
            let cases = [
                (
                    error(StatusCode::BAD_REQUEST).await,
                    hanging().await,
                    StatusCode::BAD_REQUEST,
                    true,
                ),
                (
                    error(StatusCode::INTERNAL_SERVER_ERROR).await,
                    hanging().await,
                    StatusCode::BAD_GATEWAY,
                    true,
                ),
                // Transport failure: the prefill body is cut off.
                (
                    MockWorker::start_returning_partial_body(StatusCode::OK, b"{").await,
                    hanging().await,
                    StatusCode::BAD_GATEWAY,
                    true,
                ),
                (
                    hanging().await,
                    error(StatusCode::TOO_MANY_REQUESTS).await,
                    StatusCode::TOO_MANY_REQUESTS,
                    false,
                ),
            ];
            for (prefill, decode, expected, prefill_blamed) in cases {
                let ctx = pd_ctx(&prefill.url, &decode.url, reorg);
                let response = tokio::time::timeout(
                    Duration::from_secs(1),
                    build_router(ctx.clone()).oneshot(chat(stream)),
                )
                .await
                .unwrap()
                .unwrap();
                assert_eq!(response.status(), expected);
                let body = response.into_body().collect().await.unwrap().to_bytes();
                if expected == StatusCode::BAD_GATEWAY {
                    assert!(String::from_utf8_lossy(&body).contains("prefill_failed"));
                } else {
                    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), rejected);
                }
                let (url, mode) = match prefill_blamed {
                    true => (&prefill.url, "prefill"),
                    false => (&decode.url, "decode"),
                };
                assert!(ctx.metrics.render().contains(&format!(
                    r#"worker_url="{url}",model_id="tiny",mode="{mode}""#
                )));
                let decode_worker = ctx.registry.get(&WorkerId("d".into())).unwrap();
                assert_eq!(decode_worker.router_inflight_load(), 0);
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
