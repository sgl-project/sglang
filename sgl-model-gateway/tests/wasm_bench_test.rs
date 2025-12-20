use std::{num::NonZeroUsize, time::Instant};

use lru::LruCache;
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
fn benchmark_wasm_instantiation_overhead() -> Result<()> {
    println!("\n=======================================================");
    println!("🚀 BENCHMARKING WASM INSTANCE POOLING OPTIMIZATION");
    println!("=======================================================");

    // Reduced to 1000 because "Standard" now simulates full re-compilation
    let iterations = 1000;

    // --- Scenario A: Standard Allocator (Baseline) ---
    // Simulates the unoptimized state: Re-compiling & Re-allocating every request
    let engine_standard = Engine::default();

    let start_standard = Instant::now();
    for _ in 0..iterations {
        // SIMULATE NO CACHE: Compile inside the loop
        let module_standard = Module::new(&engine_standard, WASM_WAT)?;

        let mut store = Store::new(&engine_standard, ());
        let instance = Instance::new(&mut store, &module_standard, &[])?;
        let run_func = instance.get_typed_func::<(i32, i32), i32>(&mut store, "run")?;
        let _ = run_func.call(&mut store, (10, 20))?;
    }
    let duration_standard = start_standard.elapsed();

    // --- Scenario B: Pooled Allocator (Optimized) ---
    // Simulates the optimized state: LRU Cache Hit + Pooled Memory
    let mut pool_config = PoolingAllocationConfig::default();
    pool_config.total_core_instances(100);

    let mut config = Config::new();
    config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool_config));

    let engine_pooled = Engine::new(&config)?;

    // Setup Cache (The "New" Code Logic)
    let cache_capacity = NonZeroUsize::new(100).unwrap();
    let mut cache: LruCache<Vec<u8>, Module> = LruCache::new(cache_capacity);

    // Warm up cache
    let key = WASM_WAT.as_bytes().to_vec();
    let module_precompiled = Module::new(&engine_pooled, WASM_WAT)?;
    cache.push(key.clone(), module_precompiled);

    let start_pooled = Instant::now();
    for _ in 0..iterations {
        // SIMULATE CACHE HIT: Retrieve from LRU
        let module_pooled = cache.get(&key).unwrap().clone();

        let mut store = Store::new(&engine_pooled, ());
        let instance = Instance::new(&mut store, &module_pooled, &[])?;
        let run_func = instance.get_typed_func::<(i32, i32), i32>(&mut store, "run")?;
        let _ = run_func.call(&mut store, (10, 20))?;
    }
    let duration_pooled = start_pooled.elapsed();

    // --- Calculations ---
    let speedup = duration_standard.as_secs_f64() / duration_pooled.as_secs_f64();
    let avg_std_us = (duration_standard.as_nanos() as f64 / iterations as f64) / 1000.0;
    let avg_pool_us = (duration_pooled.as_nanos() as f64 / iterations as f64) / 1000.0;

    // --- Output in OLD FORMAT ---
    println!("Iterations:           {}", iterations);
    println!("-------------------------------------------------------");
    println!("Metric                | Standard (No Pool) | Pooled (Optimized)");
    println!("----------------------|--------------------|-------------------");
    println!(
        "Total Time            | {:<18.2?} | {:<18.2?}",
        duration_standard, duration_pooled
    );
    println!(
        "Avg Latency (per op)  | {:<10.3} µs       | {:<10.3} µs",
        avg_std_us, avg_pool_us
    );
    println!("-------------------------------------------------------");
    println!("⚡ SPEEDUP FACTOR:      {:.2}x FASTER", speedup);
    println!("=======================================================\n");

    assert!(speedup > 1.5, "Pooling should be significantly faster!");
    Ok(())
}
