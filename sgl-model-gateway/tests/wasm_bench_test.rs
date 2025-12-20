use std::{num::NonZeroUsize, time::Instant};

use lru::LruCache; // Ensure 'lru' is in your dev-dependencies
use wasmtime::*;

const WASM_WAT: &str = r#"
    (module
        (memory (export "memory") 1)
        (func (export "run") (param i32 i32) (result i32)
            local.get 0
            local.get 1
            i32.add)
    )
"#;

#[test]
fn benchmark_wasm_full_pipeline_optimization() -> Result<()> {
    println!("\n=======================================================");
    println!("🚀 BENCHMARKING: CACHING + POOLING VS BASELINE");
    println!("=======================================================");

    let iterations = 1000; // Reduced iterations because "Standard" is slow!

    // --- Scenario A: The "Old" Way (No Cache + Standard Allocator) ---
    // This simulates the issue you fixed: Re-compiling and re-allocating every request.
    let engine_standard = Engine::default();

    let start_standard = Instant::now();
    for _ in 0..iterations {
        // 1. Compile (Simulating the lack of caching/re-compilation)
        let module = Module::new(&engine_standard, WASM_WAT)?;

        // 2. Instantiate (Standard allocator overhead)
        let mut store = Store::new(&engine_standard, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let run_func = instance.get_typed_func::<(i32, i32), i32>(&mut store, "run")?;
        let _ = run_func.call(&mut store, (10, 20))?;
    }
    let duration_standard = start_standard.elapsed();

    // --- Scenario B: The "New" Way (LRU Cache + Pooled Allocator) ---
    let mut pool_config = PoolingAllocationConfig::default();
    pool_config.total_core_instances(100);

    let mut config = Config::new();
    config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool_config));

    let engine_pooled = Engine::new(&config)?;

    // Initialize Cache (Simulating your new worker_loop)
    let cache_capacity = NonZeroUsize::new(100).unwrap();
    let mut cache: LruCache<Vec<u8>, Module> = LruCache::new(cache_capacity);

    // Pre-seed the cache (Simulating a "Warm" cache hit)
    let module_precompiled = Module::new(&engine_pooled, WASM_WAT)?;
    let wasm_key = WASM_WAT.as_bytes().to_vec();
    cache.push(wasm_key.clone(), module_precompiled);

    let start_pooled = Instant::now();
    for _ in 0..iterations {
        // 1. LRU Lookup (Simulating the new cache check)
        let module = cache.get(&wasm_key).expect("Cache hit").clone();

        // 2. Instantiate (Pooled allocator - fast path)
        let mut store = Store::new(&engine_pooled, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let run_func = instance.get_typed_func::<(i32, i32), i32>(&mut store, "run")?;
        let _ = run_func.call(&mut store, (10, 20))?;
    }
    let duration_pooled = start_pooled.elapsed();

    // --- Calculations ---
    let speedup = duration_standard.as_secs_f64() / duration_pooled.as_secs_f64();
    let avg_std_us = (duration_standard.as_nanos() as f64 / iterations as f64) / 1000.0;
    let avg_pool_us = (duration_pooled.as_nanos() as f64 / iterations as f64) / 1000.0;

    println!("Iterations:           {}", iterations);
    println!("-------------------------------------------------------");
    println!("Metric                | Old (Compile+Alloc) | New (LRU+Pool)");
    println!("----------------------|---------------------|----------------");
    println!(
        "Total Time            | {:<19.2?} | {:<18.2?}",
        duration_standard, duration_pooled
    );
    println!(
        "Avg Latency (per op)  | {:<10.3} µs       | {:<10.3} µs",
        avg_std_us, avg_pool_us
    );
    println!("-------------------------------------------------------");
    println!("⚡ SPEEDUP FACTOR:      {:.2}x FASTER", speedup);
    println!("=======================================================\n");

    assert!(
        speedup > 10.0,
        "The optimized pipeline should be vastly faster!"
    );
    Ok(())
}
