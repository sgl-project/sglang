use std::sync::Arc;

use axum::{
    body::Body,
    http::{HeaderMap, Request, Response, StatusCode},
    middleware,
    response::IntoResponse,
};
use criterion::{criterion_group, criterion_main, Criterion};
use http_body_util::BodyExt;
use smg::{
    app_context::AppContext, config::RouterConfig, middleware::wasm_middleware,
    protocols::chat::ChatCompletionRequest, routers::RouterTrait, server::AppState,
};
use tokio::runtime::Runtime;
use tower::{Layer, Service};

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
    ) -> Response<Body> {
        StatusCode::OK.into_response()
    }
    fn router_type(&self) -> &'static str {
        "mock"
    }
}

/// Mock service that simulates a streaming response with a 500ms delay.
async fn mock_next_streaming(_req: Request<Body>) -> Response<Body> {
    let (tx, rx) = tokio::sync::mpsc::channel(16);

    tokio::spawn(async move {
        // Send first chunk immediately
        let _ = tx
            .send(Ok::<_, std::io::Error>(bytes::Bytes::from("chunk 1 ")))
            .await;
        // Simulate generation delay
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

    // Setup AppContext with WASM enabled
    let config = RouterConfig::builder().enable_wasm(true).build_unchecked();

    let context = rt.block_on(AppContext::from_config(config, 30)).unwrap();
    let app_state = Arc::new(AppState {
        router: Arc::new(MockRouter),
        context: Arc::new(context),
        concurrency_queue_tx: None,
        router_manager: None,
        mesh_handler: None,
        mesh_sync_manager: None,
    });

    c.bench_function("wasm_middleware_pre_fix_latency", |b| {
        b.iter(|| {
            rt.block_on(async {
                let req = Request::builder()
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap();

                // Create the service by applying the middleware layer to the mock streamer
                let mut service =
                    middleware::from_fn_with_state(app_state.clone(), wasm_middleware).layer(
                        tower::service_fn(|req: Request<Body>| async move {
                            Ok::<_, std::convert::Infallible>(mock_next_streaming(req).await)
                        }),
                    );

                // Explicitly poll the service
                let response: Response<Body> =
                    service.call(req).await.expect("Middleware service failed");

                // Measure how long it takes to receive the FIRST frame
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
