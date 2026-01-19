use std::borrow::Cow;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smg::wasm::{
    config::WasmRuntimeConfig,
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    runtime::WasmRuntime,
    spec::sgl::model_gateway::middleware_types::Request,
    types::WasmComponentInput,
};
use tokio::runtime::Runtime;
use wasm_encoder::Component;

fn bench_wasm_cache_lookup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // 1. Create a valid but "heavy" WASM component (padded to ~1MB)
    // We use Cow::Owned and Cow::Borrowed to satisfy wasm-encoder types
    let mut encoder = Component::new();
    encoder.section(&wasm_encoder::CustomSection {
        name: Cow::Borrowed("padding"),
        data: Cow::Owned(vec![0u8; 1_000_000]),
    });
    let wasm_bytes = encoder.finish();

    let config = WasmRuntimeConfig::default();
    let runtime = WasmRuntime::new(config).unwrap();

    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

    // 2. Manually construct the Request object
    let request = Request {
        method: "GET".to_string(),
        path: "/test".to_string(),
        query: "".to_string(),
        headers: vec![],
        body: vec![],
        request_id: "bench-id".to_string(),
        now_epoch_ms: 1700000000,
    };
    let input = WasmComponentInput::MiddlewareRequest(request);

    // Pre-warm the cache so the benchmark only measures hit latency
    rt.block_on(runtime.execute_component_async(
        wasm_bytes.clone(),
        attach_point.clone(),
        input.clone(),
    ))
    .unwrap();

    c.bench_function("wasm_cache_lookup_hot", |b| {
        b.iter(|| {
            rt.block_on(runtime.execute_component_async(
                black_box(wasm_bytes.clone()),
                black_box(attach_point.clone()),
                black_box(input.clone()),
            ))
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100);
    targets = bench_wasm_cache_lookup
}
criterion_main!(benches);
