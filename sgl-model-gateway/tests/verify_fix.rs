use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use sgl_model_gateway::tokenizer::{
    huggingface::HuggingFaceTokenizer,
    traits::Encoder,
};

async fn latency_monitor(name: &str) -> Vec<Duration> {
    let mut spikes = Vec::new();
    let start = Instant::now();
    let mut last_tick = start;

    loop {
        // Sleep for 10ms. If the runtime is healthy, we wake up in ~10-12ms.
        sleep(Duration::from_millis(10)).await;

        let now = Instant::now();
        let elapsed = now.duration_since(last_tick);

        // If a "10ms sleep" took > 30ms, something blocked the thread.
        if elapsed.as_millis() > 30 {
            println!("[{}] Spike detected: {:?}", name, elapsed);
            spikes.push(elapsed);
        }
        last_tick = now;

        // Run for a bit longer than the heavy task
        if start.elapsed() > Duration::from_millis(500) {
            break;
        }
    }
    spikes
}

// 1. SCENARIO A: The "Before" case (Problematic)
// This calls encode() directly on the async thread.
async fn run_blocking_scenario(tokenizer: Arc<HuggingFaceTokenizer>, text: String) -> bool {
    println!("\n--- Running SCENARIO A: Direct Blocking Call ---");
    let monitor = tokio::spawn(latency_monitor("Blocking"));

    // YIELD to ensure monitor starts running before we block
    tokio::task::yield_now().await;
    sleep(Duration::from_millis(10)).await;

    let start = Instant::now();
    // !!! The Problem: Calling CPU intensive task on async thread !!!
    let _ = tokenizer.encode(&text).unwrap();
    println!("Blocking Task took: {:?}", start.elapsed());

    let spikes = monitor.await.unwrap();
    println!("Scenario A Spikes: {}", spikes.len());
    !spikes.is_empty()
}

// 2. SCENARIO B: The "After" case (Fixed)
// This matches your fix in preparation.rs using spawn_blocking
async fn run_fixed_scenario(tokenizer: Arc<HuggingFaceTokenizer>, text: String) -> bool {
    println!("\n--- Running SCENARIO B: spawn_blocking (Fixed) ---");
    let monitor = tokio::spawn(latency_monitor("Fixed"));

    // YIELD to ensure monitor starts running
    tokio::task::yield_now().await;
    sleep(Duration::from_millis(10)).await;

    let start = Instant::now();

    // !!! The Fix: Wrapping CPU task in spawn_blocking !!!
    let _ = tokio::task::spawn_blocking(move || {
        tokenizer.encode(&text).unwrap()
    }).await.unwrap();

    println!("Fixed Task took: {:?}", start.elapsed());

    let spikes = monitor.await.unwrap();
    println!("Scenario B Spikes: {}", spikes.len());
    !spikes.is_empty()
}

#[tokio::test(flavor = "current_thread")]
async fn test_compare_blocking_vs_fixed() {
    let tokenizer_path = "tests/tokenizer.json";
    if !std::path::Path::new(tokenizer_path).exists() {
        println!("Skipping: tokenizer.json not found");
        return;
    }

    let tokenizer = Arc::new(HuggingFaceTokenizer::from_file(tokenizer_path).unwrap());
    // Create text large enough to cause ~200ms delay
    let heavy_text = "The quick brown fox jumps over the lazy dog. ".repeat(10_000);

    // Run "Before"
    let had_spikes_blocking = run_blocking_scenario(tokenizer.clone(), heavy_text.clone()).await;

    // Run "After"
    let had_spikes_fixed = run_fixed_scenario(tokenizer.clone(), heavy_text).await;

    // Assertions
    if had_spikes_blocking {
        println!("\nVERDICT: Blocking scenario correctly caused latency spikes.");
    } else {
        println!("\nWARNING: Blocking scenario was too fast to cause spikes. Increase text size.");
    }

    if !had_spikes_fixed {
        println!("VERDICT: Fixed scenario kept the event loop responsive (No spikes).");
    } else {
        println!("FAILURE: Fixed scenario still caused spikes.");
    }

    assert!(had_spikes_blocking, "Expected blocking code to cause spikes");
    assert!(!had_spikes_fixed, "Expected fixed code to NOT cause spikes");
}
