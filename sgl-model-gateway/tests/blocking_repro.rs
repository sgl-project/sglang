use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use sgl_model_gateway::tokenizer::{
    huggingface::HuggingFaceTokenizer,
    traits::Encoder, // <--- FIX: Imported the Encoder trait here
};

// 1. Setup a "Monitor" task that expects to run every 10ms
// If it takes > 30ms (allow some buffer), we consider it "blocked".
async fn latency_monitor() -> Vec<Duration> {
    let mut spikes = Vec::new();
    let start = Instant::now();
    let mut last_tick = start;

    loop {
        // Yield to let other tasks run
        tokio::task::yield_now().await;
        sleep(Duration::from_millis(10)).await;

        let now = Instant::now();
        let elapsed = now.duration_since(last_tick);

        // If the gap between ticks is significantly larger than sleep time (e.g., >30ms),
        // it means the thread was blocked by something else.
        if elapsed.as_millis() > 30 {
            spikes.push(elapsed);
        }
        last_tick = now;

        if start.elapsed() > Duration::from_secs(2) {
            break;
        }
    }
    spikes
}

#[tokio::test(flavor = "current_thread")]
async fn test_reproduce_cpu_blocking() {
    // A. Setup Tokenizer (Simulate heavy load)
    // We reuse the logic from your existing benchmarks to get a valid tokenizer
    // !!! IMPORTANT: Replace this path with a valid tokenizer.json on your system !!!
    // Example: "models/Meta-Llama-3-8B-Instruct/tokenizer.json"
    let tokenizer_path = "tests/tokenizer.json";

    // Fallback check to avoid crashing if file doesn't exist during dry run
    if !std::path::Path::new(tokenizer_path).exists() {
        println!("SKIPPING TEST: '{}' not found. Please update path in tests/blocking_repro.rs", tokenizer_path);
        return;
    }

    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path)
            .expect("Failed to load tokenizer")
    );

    // B. Create a HUGE input text (~50k+ tokens equivalent)
    let heavy_text = "The quick brown fox jumps over the lazy dog. ".repeat(10_000);

    // C. Spawn the Latency Monitor
    let monitor_handle = tokio::spawn(latency_monitor());

    // D. Run the Blocking Tokenization Task DIRECTLY in the async runtime
    let tokenizer_clone = tokenizer.clone();
    let start_block = Instant::now();

    println!("Starting blocking encoding...");

    // !!! PROBLEM: blocking call in async fn !!!
    // Because we imported 'Encoder', this method is now available
    let _encoding = tokenizer_clone.encode(&heavy_text).unwrap();

    let block_duration = start_block.elapsed();
    println!("Tokenization took: {:?}", block_duration);

    // E. Check Results
    let spikes = monitor_handle.await.unwrap();

    println!("Detected latency spikes: {:?}", spikes);

    if block_duration.as_millis() > 50 {
        if spikes.is_empty() {
             println!("WARNING: Operation took {:?} but no spikes detected. Runtime might be multithreaded enough or load too light.", block_duration);
        } else {
             println!("SUCCESS: Reproduced CPU blocking. Monitor was starved for {:?}", spikes[0]);
        }
    } else {
        println!("WARNING: Tokenization was too fast ({:?}) to block significantly. Increase text size.", block_duration);
    }
}
