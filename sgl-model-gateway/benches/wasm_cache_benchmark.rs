use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smg::wasm::{
    config::WasmRuntimeConfig,
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    runtime::WasmRuntime,
    types::WasmComponentInput,
};
use tokio::runtime::Runtime;

fn bench_wasm_cache_lookup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // 1. Create a 1MB WASM component (minimal valid WAT + padding)
    // Large enough to make byte-by-byte comparison overhead measurable.
    let mut wasm_bytes = wat::parse_str(
        r#"
        (component
            (import "sgl:model-gateway/middleware-types" (instance $types))
            (export "sgl:model-gateway/middleware-on-request" (instance 0))
        )
    "#,
    )
    .unwrap();
    wasm_bytes.extend(vec![0u8; 1_000_000]); // Pad to ~1MB

    let config = WasmRuntimeConfig::default();
    let runtime = WasmRuntime::new(config).unwrap();

    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

    // Create a dummy input (structure depends on your actual WasmComponentInput)
    // We use a dummy here as we expect the benchmark to hit the cache comparison logic.
    let input = WasmComponentInput::MiddlewareRequest(Default::default());

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
    config = Criterion::default().sample_size(50);
    targets = bench_cache_lookup
}
criterion_main!(benches);
