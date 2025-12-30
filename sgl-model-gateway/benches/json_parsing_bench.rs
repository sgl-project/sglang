use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::Deserialize;
use serde_json::Value;

// --- 1. SAMPLE DATA GENERATOR ---

// A realistic worker status object with mixed types (strings, numbers, objects)
// We only care about `num_tokens`, but the parser has to wade through the rest.
const ITEM_TEMPLATE: &str = r#"{
    "id": "req_0000000",
    "status": "running",
    "num_tokens": 450,
    "logprob": -0.123456789,
    "generated_text": "This is some random text that represents the output...",
    "finish_reason": null,
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 400
    },
    "metadata": {
        "user_id": "user_123",
        "session_id": "sess_abc",
        "trace_id": "trace_xyz"
    }
}"#;

fn generate_json(count: usize) -> String {
    let item: Value = serde_json::from_str(ITEM_TEMPLATE).unwrap();
    let mut list = Vec::with_capacity(count);
    for _ in 0..count {
        list.push(item.clone());
    }
    serde_json::to_string(&list).unwrap()
}

// --- 2. THE "BEFORE" IMPLEMENTATION (Slow) ---
// Mimics sgl-model-gateway/src/core/worker_manager.rs
// Parses everything into a Heap-allocated Value tree, then iterates.
fn parse_dynamic_value(json_str: &str) -> isize {
    let parsed: Result<Value, _> = serde_json::from_str(json_str);
    match parsed {
        Ok(json) => {
            if let Some(arr) = json.as_array() {
                arr.iter()
                    .filter_map(|e| {
                        // Inefficient: String hashing ("num_tokens") + HashMap lookup per item
                        e.get("num_tokens").and_then(|v| v.as_i64())
                    })
                    .sum::<i64>() as isize
            } else {
                0
            }
        }
        Err(_) => 0,
    }
}

// --- 3. THE "AFTER" IMPLEMENTATION (Fast) ---
// Uses strict typing. Serde will skip "metadata", "usage", "text", etc.
// This results in almost ZERO heap allocations for the data itself.
#[derive(Deserialize)]
struct LoadItem {
    num_tokens: i64,
}

fn parse_typed_struct(json_str: &str) -> isize {
    let parsed: Result<Vec<LoadItem>, _> = serde_json::from_str(json_str);
    match parsed {
        Ok(items) => items.iter().map(|i| i.num_tokens).sum::<i64>() as isize,
        Err(_) => 0,
    }
}

// --- 4. BENCHMARK SUITE ---
fn bench_load_monitor_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_monitor_json_parsing");

    // We test multiple sizes to see how the overhead scales
    // 10 items = Small worker load
    // 1000 items = High load / Batch processing
    let sizes = [10, 100, 1000];

    for &size in &sizes {
        let json_data = generate_json(size);

        group.throughput(Throughput::Bytes(json_data.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("dynamic_value (before)", size),
            &json_data,
            |b, data| b.iter(|| parse_dynamic_value(black_box(data))),
        );

        group.bench_with_input(
            BenchmarkId::new("typed_struct (after)", size),
            &json_data,
            |b, data| b.iter(|| parse_typed_struct(black_box(data))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_load_monitor_parsing);
criterion_main!(benches);
