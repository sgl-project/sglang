use std::sync::Arc;

use axum::{
    body::Body,
    http::{HeaderMap, Request, Response, StatusCode},
    middleware,
};
use criterion::{criterion_group, criterion_main, Criterion};
use futures::StreamExt;
use http_body_util::BodyExt;
use smg::{
    app_context::AppContext, config::RouterConfig, middleware::wasm_middleware,
    protocols::chat::ChatCompletionRequest, routers::RouterTrait, server::AppState,
};
use tokio::runtime::Runtime;
use tower::{Service, ServiceExt};

/// Dummy router to satisfy AppState requirements for the benchmark
#[derive(Debug)]
struct MockRouter;

#[async_trait::async_trait]
impl RouterTrait for MockRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        StatusCode::OK.into_response()
    }
    fn router_type(&self) -> &'static str {
        "mock"
    }
}

/// Mock service that simulates a streaming response with a 500ms delay.
/// This delay represents the worker's generation time.
async fn mock_next_streaming(_req: Request<Body>) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel(16);

    tokio::spawn(async move {
        // Send first chunk immediately
        let _ = tx
            .send(Ok::<_, std::io::Error>(bytes::Bytes::from("chunk 1 ")))
            .await;
        // Simulate generation delay
        // IN THE PRE-FIX STATE: the middleware will block here for 500ms!
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        // Send final chunk
        let _ = tx
            .send(Ok::<_, std::io::Error>(bytes::Bytes::from("chunk 2")))
            .await;
    });

    Response::new(Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(rx),
    ))
}

fn bench_wasm_middleware_buffering(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Setup AppContext with WASM enabled (required to hit the buffering path)
    let config = RouterConfig::builder().enable_wasm(true).build_unchecked();

    let context = rt.block_on(AppContext::from_config(config, 30)).unwrap();
    let app_state = Arc::new(AppState {
        router: Arc::new(MockRouter),
        context: Arc::new(context),
        concurrency_queue_tx: None,
        router_manager: None,
    });

    c.bench_function("wasm_middleware_pre_fix_latency", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap();

                // Build a Tower service applying the middleware to our mock streamer
                let mut service =
                    middleware::from_fn_with_state(app_state.clone(), wasm_middleware).layer(
                        tower::service_fn(|req: Request<Body>| async move {
                            Ok::<_, std::convert::Infallible>(mock_next_streaming(req).await)
                        }),
                    );

                // Call the service and wait for the response header
                let response = service.call(req).await.unwrap();

                // Measure how long it takes to receive the FIRST frame (chunk)
                let mut body = response.into_body();
                let _first_frame = body.frame().await;
            });
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_wasm_middleware_buffering
}
criterion_main!(benches);
