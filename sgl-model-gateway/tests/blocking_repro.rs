use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use sgl_model_gateway::tokenizer::{
    huggingface::HuggingFaceTokenizer,
    traits::Tokenizer,
};

// 1. Setup a "Monitor" task that expects to run every 10ms
// If it takes > 20ms, we consider it "blocked".
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

        // If the gap between ticks is significantly larger than sleep time (e.g., >20ms),
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

#[tokio::test]
async fn test_reproduce_cpu_blocking() {
    // A. Setup Tokenizer (Simulate heavy load)
    // We reuse the logic from your existing benchmarks to get a valid tokenizer
    // Note: You might need to adjust the path to a real model file or download one
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file("path/to/your/tokenizer.json")
            .expect("Failed to load tokenizer")
    );

    // B. Create a HUGE input text (50k+ tokens)
    let heavy_text = "The quick brown fox jumps over the lazy dog. ".repeat(10_000);

    // C. Spawn the Latency Monitor
    let monitor_handle = tokio::spawn(latency_monitor());

    // D. Run the Blocking Tokenization Task DIRECTLY in the async runtime
    // This replicates the problematic code in `prepare_generate`
    let tokenizer_clone = tokenizer.clone();
    let start_block = Instant::now();

    // !!! PROBLEM: blocking call in async fn !!!
    let _encoding = tokenizer_clone.encode(&heavy_text).unwrap();

    let block_duration = start_block.elapsed();
    println!("Tokenization took: {:?}", block_duration);

    // E. Check Results
    let spikes = monitor_handle.await.unwrap();

    println!("Detected latency spikes: {:?}", spikes);

    // If tokenization took > 50ms, we expect spikes.
    // If the runtime was blocked, the monitor couldn't wake up on time.
    if block_duration.as_millis() > 50 {
        assert!(!spikes.is_empty(), "Expected latency spikes due to CPU blocking, but found none!");
        println!("SUCCESS: Reproduced CPU blocking. Monitor was starved for {:?}", spikes[0]);
    } else {
        println!("WARNING: Tokenization was too fast to block significantly. Increase text size.");
    }
}
