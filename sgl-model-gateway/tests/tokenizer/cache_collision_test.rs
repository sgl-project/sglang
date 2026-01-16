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

    // First call populates cache with special tokens
    let enc1 = cached.encode(input, true).expect("First encode failed");
    let tokens_with_special = enc1.token_ids().to_vec();

    // Second call should return tokens WITHOUT special tokens
    let enc2 = cached.encode(input, false).expect("Second encode failed");
    let tokens_without_special = enc2.token_ids().to_vec();

    assert_ne!(
        tokens_with_special, tokens_without_special,
        "BUG: L0 cache returned the same result for different add_special_tokens values!"
    );
}
