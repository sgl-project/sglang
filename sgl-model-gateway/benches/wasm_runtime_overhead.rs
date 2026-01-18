use criterion::{criterion_group, criterion_main, Criterion};
use smg::wasm::{
    config::WasmRuntimeConfig,
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    runtime::WasmRuntime,
    spec::sgl::model_gateway::middleware_types::Request,
    types::WasmComponentInput,
};
use tokio::runtime::Runtime;


const AUTH_WASM: &[u8] = include_bytes!(
    "../examples/wasm/wasm-guest-auth/target/wasm32-wasip2/release/wasm_guest_auth.component.wasm"
);

fn bench_wasm_runtime_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = WasmRuntimeConfig::default();
    let runtime = WasmRuntime::new(config).unwrap();

    let request = Request {
        method: "GET".to_string(),
        path: "/v1/chat/completions".to_string(),
        query: "".to_string(),
        headers: vec![],
        body: vec![],
        request_id: "bench-id".to_string(),
        now_epoch_ms: 0,
    };

    let input = WasmComponentInput::MiddlewareRequest(request);
    let attach_point = WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

    c.bench_function("wasm_runtime_overhead_pre_fix", |b| {

        b.iter(|| {
            rt.block_on(async {
                runtime
                    .execute_component_async(
                        AUTH_WASM.to_vec(),
                        attach_point.clone(),
                        input.clone(),
                    )
                    .await
                    .expect("WASM execution failed");
            })
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_wasm_runtime_overhead
}
criterion_main!(benches);
