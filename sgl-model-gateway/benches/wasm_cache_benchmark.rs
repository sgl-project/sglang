use std::borrow::Cow;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sha2::{Digest, Sha256};
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

    // 1. Create a 1MB WASM binary.
    // This is valid enough to pass the wasmtime compilation check and enter the cache,
    // even though it will fail the later instantiation step.
    let mut encoder = Component::new();
    encoder.section(&wasm_encoder::CustomSection {
        name: Cow::Borrowed("padding"),
        data: Cow::Owned(vec![0u8; 1_000_000]),
    });
    let wasm_bytes = encoder.finish();

    let mut hasher = sha2::Sha256::new();
    hasher.update(&wasm_bytes);
    let wasm_hash: [u8; 32] = hasher.finalize().into();

    let config = WasmRuntimeConfig::default();
    let runtime = WasmRuntime::new(config).unwrap();

    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

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

    // 3. Pre-warm the cache.
    // This will successfully compile the binary and add it to the LRU cache
    // inside the worker thread before the first benchmark iteration.
    let _ = rt.block_on(runtime.execute_component_async(
        wasm_bytes.clone(),
        wasm_hash,
        attach_point.clone(),
        input.clone(),
    ));

    c.bench_function("wasm_cache_lookup_hot", |b| {
        b.iter(|| {
            // We measure the full async roundtrip.
            // On every iteration, the worker will perform an O(N) byte comparison
            // of the 1MB key in its LruCache.
            // We ignore the Result because the inevitable instantiation error
            // happens after the cache lookup we are targeting.
            let _ = rt.block_on(runtime.execute_component_async(
                black_box(wasm_bytes.clone()),
                black_box(wasm_hash),
                black_box(attach_point.clone()),
                black_box(input.clone()),
            ));
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100);
    targets = bench_wasm_cache_lookup
}
criterion_main!(benches);
