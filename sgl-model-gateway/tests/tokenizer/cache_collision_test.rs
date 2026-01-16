use std::sync::Arc;

use smg::tokenizer::{
    cache::{CacheConfig, CachedTokenizer},
    hub::download_tokenizer_from_hf,
    huggingface::HuggingFaceTokenizer,
    traits::Encoder,
};

#[tokio::test]
async fn test_l0_cache_key_collision() {
    let cache_dir = download_tokenizer_from_hf("Qwen/Qwen3-4B-Instruct-2507")
        .await
        .expect("Failed to download tokenizer");
    let tokenizer_path = cache_dir.join("tokenizer.json");

    let inner = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load inner tokenizer"),
    );

    let config = CacheConfig {
        enable_l0: true,
        l0_max_entries: 100,
        enable_l1: false,
        l1_max_memory: 0,
    };
    let cached = CachedTokenizer::new(inner.clone(), config);
    let input = "Hello";

    // This creates a cache entry for ("Hello", true)
    let _ = cached.encode(input, true).expect("First encode failed");

    // add_special_tokens = false
    // This SHOULD BE A MISS because the flag is different.
    let _ = cached.encode(input, false).expect("Second encode failed");

    // 4. Verify via Cache Stats
    let stats = cached.cache_stats().expect("Stats should be available");

    println!("Cache Stats: Hits={}, Misses={}", stats.hits, stats.misses);

    assert_eq!(
        stats.hits, 0,
        "BUG: L0 cache HIT on the second call! This means it ignored the add_special_tokens flag."
    );
    assert_eq!(
        stats.misses, 2,
        "Expected 2 misses (one for each unique flag value)."
    );

    // Third call: add_special_tokens = true (Repeat of first call)
    // This SHOULD BE A HIT.
    let _ = cached.encode(input, true).expect("Third encode failed");
    let stats_after = cached.cache_stats().expect("Stats should be available");

    assert_eq!(
        stats_after.hits, 1,
        "Expected exactly 1 hit for the repeated call"
    );
}
