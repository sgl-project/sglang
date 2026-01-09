use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, Response, StatusCode},
    middleware::Next,
};
use criterion::{criterion_group, criterion_main, Criterion};
use futures::StreamExt;
use smg::{
    app_context::AppContext, config::RouterConfig, middleware::wasm_middleware, server::AppState,
};
use tokio::runtime::Runtime;

// Mock the 'Next' service to return a streaming body that takes time to complete
async fn mock_next_streaming(req: Request<Body>) -> Response<Body> {
    let (mut tx, body) = Body::channel();

    tokio::spawn(async move {
        // Send first chunk immediately
        tx.send_data("chunk 1 ".into()).await.unwrap();
        // Simulate generation delay
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        // Send final chunk
        tx.send_data("chunk 2".into()).await.unwrap();
    });

    Response::new(body)
}

fn bench_wasm_middleware_buffering(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Setup minimal AppState with WASM enabled (or mock it to trigger the code path)
    let config = RouterConfig::default(); // Adjust based on actual config struct
    let context = rt.block_on(AppContext::from_config(config, 30)).unwrap();
    let app_state = Arc::new(AppState {
        router: Arc::new(smg::routers::mock::MockRouter::new()),
        context: Arc::new(context),
        concurrency_queue_tx: None,
        router_manager: None,
    });

    c.bench_function("wasm_middleware_pre_fix_latency", |b| {
        b.to_async(&rt).iter(|| async {
            let req = Request::builder()
                .uri("/v1/chat/completions")
                .body(Body::empty())
                .unwrap();

            // Run the middleware
            // In the pre-fix state, this .await will wait for the 500ms delay
            // inside wasm_middleware because of axum::body::to_bytes
            let response = wasm_middleware(
                axum::extract::State(app_state.clone()),
                req,
                Next::from_fn(mock_next_streaming),
            )
            .await
            .unwrap();

            // Measure how long it takes to get the FIRST chunk
            let mut stream = response.into_body().into_data_stream();
            let _first_chunk = stream.next().await;
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10); // Low sample size as each run takes >500ms
    targets = bench_wasm_middleware_buffering
}
criterion_main!(benches);
