use criterion::{criterion_group, criterion_main, Criterion};
use smg::wasm::{
    config::WasmRuntimeConfig,
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    runtime::WasmRuntime,
    types::{MiddlewareRequest, WasmComponentInput},
};
use tokio::runtime::Runtime;

// A minimal valid WASM component binary (pre-compiled)
// that satisfies the SglModelGateway interface.
// For the purpose of this benchmark, we can use a placeholder
// or ensure we have a valid component for the runtime to "instantiate".
// Note: In a real environment, you'd load a .wasm file built from your guest examples.
const DUMMY_WASM: &[u8] = include_bytes!("../examples/wasm/wasm-guest-auth/target/wasm32-wasip2/release/wasm_guest_auth.component.wasm");

fn bench_wasm_runtime_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = WasmRuntimeConfig::default();
    let runtime = WasmRuntime::new(config).unwrap();

    let input = WasmComponentInput::MiddlewareRequest(MiddlewareRequest {
        method: "GET".to_string(),
        uri: "/test".to_string(),
        headers: vec![],
        body: vec![],
    });

    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

    c.bench_function("wasm_runtime_execution_latency", |b| {
        b.to_async(&rt).iter(|| async {
            runtime
                .execute_component_async(DUMMY_WASM.to_vec(), attach_point.clone(), input.clone())
                .await
                .expect("WASM execution failed");
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(50); // Larger sample for better stats
    targets = bench_wasm_runtime_execution
}
criterion_main!(benches);
