use super::*;
use crate::common::mock_worker::MockWorker;
use futures::StreamExt;
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

/// A prefill worker that answers `status` once `release` is notified.
async fn start_prefill(release: Arc<tokio::sync::Notify>, status: StatusCode) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move || {
            let release = release.clone();
            async move {
                release.notified().await;
                (status, axum::Json(json!({})))
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

async fn wait_until(mut done: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while !done() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

/// Request outcomes recorded against the worker at `url`.
fn outcomes(ctx: &AppContext, url: &str, mode: &str) -> Vec<String> {
    let prefix = format!(
        r#"sgl_router_worker_requests_total{{worker_url="{url}",model_id="tiny",mode="{mode}",outcome=""#
    );
    let metrics = ctx.metrics.render();
    metrics
        .lines()
        .filter_map(|line| Some(line.strip_prefix(&prefix)?.split('"').next()?.to_string()))
        .collect()
}

/// Either side's failure ends the request without waiting for the other side.
#[tokio::test]
async fn pd_failure_returns_without_waiting_for_the_other_side() {
    let rejected = json!({"error": "rejected"});
    let error = |status| MockWorker::start_returning_error(status, rejected.clone());
    let hanging = || MockWorker::start_hanging(Duration::from_secs(10));
    for reorg in [false, true] {
        for stream in [false, true] {
            // (prefill, decode, client status, router error code, whether prefill is blamed)
            let cases = [
                (
                    error(StatusCode::BAD_REQUEST).await,
                    hanging().await,
                    StatusCode::BAD_REQUEST,
                    None,
                    true,
                ),
                (
                    error(StatusCode::SERVICE_UNAVAILABLE).await,
                    hanging().await,
                    StatusCode::SERVICE_UNAVAILABLE,
                    None,
                    true,
                ),
                (
                    error(StatusCode::INTERNAL_SERVER_ERROR).await,
                    hanging().await,
                    StatusCode::BAD_GATEWAY,
                    Some("prefill_failed"),
                    true,
                ),
                // Transport failure: the prefill body is cut off.
                (
                    MockWorker::start_returning_partial_body(StatusCode::OK, b"{").await,
                    hanging().await,
                    StatusCode::BAD_GATEWAY,
                    Some("upstream_body_incomplete"),
                    true,
                ),
                (
                    hanging().await,
                    error(StatusCode::TOO_MANY_REQUESTS).await,
                    StatusCode::TOO_MANY_REQUESTS,
                    None,
                    false,
                ),
            ];
            for (prefill, decode, expected, code, prefill_blamed) in cases {
                let ctx = pd_ctx(&prefill.url, &decode.url, reorg);
                let response = tokio::time::timeout(
                    Duration::from_secs(1),
                    build_router(ctx.clone()).oneshot(chat(stream)),
                )
                .await
                .unwrap()
                .unwrap();
                assert_eq!(response.status(), expected);
                let router_code = response.headers().get("x-router-error-code").cloned();
                assert_eq!(router_code.as_ref().map(|c| c.to_str().unwrap()), code);
                let body = response.into_body().collect().await.unwrap().to_bytes();
                if code.is_none() {
                    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), rejected);
                }
                let (blamed, other) = match prefill_blamed {
                    true => ((&prefill.url, "prefill"), (&decode.url, "decode")),
                    false => ((&decode.url, "decode"), (&prefill.url, "prefill")),
                };
                assert_eq!(outcomes(&ctx, blamed.0, blamed.1).len(), 1);
                assert!(outcomes(&ctx, other.0, other.1).is_empty());
                let decode_worker = ctx.registry.get(&WorkerId("d".into())).unwrap();
                assert_eq!(decode_worker.router_inflight_load(), 0);
            }
        }
    }
}

/// Until decode streams its first token, a prefill failure aborts decode: before
/// decode's headers the client sees prefill's error, after them the stream breaks.
#[tokio::test]
async fn prefill_failure_aborts_decode_before_its_first_token() {
    for reorg in [false, true] {
        for early_headers in [false, true] {
            let decode = if early_headers {
                MockWorker::start_slow_stream(vec!["data: late\n\n"], Duration::from_secs(10)).await
            } else {
                MockWorker::start_hanging(Duration::from_secs(10)).await
            };
            let release = Arc::new(tokio::sync::Notify::new());
            let prefill_url = start_prefill(release.clone(), StatusCode::BAD_REQUEST).await;
            let ctx = pd_ctx(&prefill_url, &decode.url, reorg);
            let request = tokio::spawn(build_router(ctx.clone()).oneshot(chat(early_headers)));
            wait_until(|| decode.captured.lock().unwrap().last_body.is_some()).await;
            if early_headers {
                let response = request.await.unwrap().unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                release.notify_one();
                assert!(response.into_body().collect().await.is_err());
                assert!(ctx.metrics.render().contains(&format!(
                    r#"sgl_router_stream_outcome_total{{worker_url="{}",model_id="tiny",outcome="aborted"}} 1"#,
                    decode.url
                )));
            } else {
                release.notify_one();
                let response = request.await.unwrap().unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            }
            let workers = ctx.registry.workers_for(&ModelId("tiny".into()));
            wait_until(|| {
                !decode.abort_log.lock().unwrap().is_empty()
                    && workers.iter().all(|w| w.router_inflight_load() == 0)
            })
            .await;
            let sent = parse_body(decode.captured.lock().unwrap().last_body.as_ref().unwrap());
            assert_eq!(
                decode.abort_log.lock().unwrap()[0],
                json!({"rid": sent["rid"], "abort_all": false})
            );
        }
    }
}

/// Decode's headers do not wait for prefill, and once decode streams its first
/// token KV transfer is done, so a later prefill failure leaves the stream intact.
#[tokio::test]
async fn prefill_failure_after_decode_first_token_keeps_the_stream() {
    for reorg in [false, true] {
        let decode = MockWorker::start_slow_stream(
            vec!["data: a\n\n", "data: [DONE]\n\n"],
            Duration::from_millis(100),
        )
        .await;
        let release = Arc::new(tokio::sync::Notify::new());
        let prefill_url = start_prefill(release.clone(), StatusCode::INTERNAL_SERVER_ERROR).await;
        let ctx = pd_ctx(&prefill_url, &decode.url, reorg);
        let response = build_router(ctx.clone()).oneshot(chat(true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body().into_data_stream();
        assert_eq!(body.next().await.unwrap().unwrap(), "data: a\n\n");
        release.notify_one();
        let prefill = ctx.registry.get(&WorkerId("p".into())).unwrap();
        wait_until(|| prefill.router_inflight_load() == 0).await;
        assert_eq!(body.next().await.unwrap().unwrap(), "data: [DONE]\n\n");
        assert!(body.next().await.is_none());
        assert!(decode.abort_log.lock().unwrap().is_empty());
        assert_eq!(outcomes(&ctx, &prefill_url, "prefill"), ["error"]);
        assert_eq!(outcomes(&ctx, &decode.url, "decode"), ["success"]);
    }
}

/// Prefill must finish KV transfer on its own, so decode failing first leaves it running.
#[tokio::test]
async fn decode_failure_leaves_prefill_running() {
    for reorg in [false, true] {
        let release = Arc::new(tokio::sync::Notify::new());
        let prefill_url = start_prefill(release.clone(), StatusCode::OK).await;
        let decode = MockWorker::start_returning_error(StatusCode::BAD_REQUEST, json!({})).await;
        let ctx = pd_ctx(&prefill_url, &decode.url, reorg);
        let response = build_router(ctx.clone())
            .oneshot(chat(false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let prefill = ctx.registry.get(&WorkerId("p".into())).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(prefill.router_inflight_load(), 1);
        release.notify_one();
        wait_until(|| prefill.router_inflight_load() == 0).await;
        assert_eq!(outcomes(&ctx, &prefill_url, "prefill"), ["success"]);
    }
}

/// A streaming prefill that fails after committing a 200 is still a prefill failure.
#[tokio::test]
async fn prefill_sse_error_event_is_a_prefill_failure() {
    for reorg in [false, true] {
        let prefill =
            MockWorker::start(vec!["data: {\"error\": {\"message\": \"boom\"}}\n\n"]).await;
        let decode = MockWorker::start_hanging(Duration::from_secs(10)).await;
        let ctx = pd_ctx(&prefill.url, &decode.url, reorg);
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            build_router(ctx.clone()).oneshot(chat(true)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(response.headers()["x-router-error-code"], "prefill_failed");
        assert_eq!(outcomes(&ctx, &prefill.url, "prefill"), ["error"]);
    }
}
