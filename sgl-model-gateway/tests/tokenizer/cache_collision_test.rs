use std::sync::Arc;
use smg::tokenizer::{
    cache::{CacheConfig, CachedTokenizer},
    hub::download_tokenizer_from_hf,
    huggingface::HuggingFaceTokenizer,
    traits::Encoder,
};

#[tokio::test]
async fn test_l0_cache_key_collision() {
    // 1. Setup the real Qwen tokenizer
    // Integration tests download this to a local cache directory
    let cache_dir = download_tokenizer_from_hf("Qwen/Qwen3-4B-Instruct-2507")
        .await
        .expect("Failed to download tokenizer");
    let tokenizer_path = cache_dir.join("tokenizer.json");

    let inner = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load inner tokenizer")
    );

    // 2. Create CachedTokenizer with L0 enabled
    let config = CacheConfig {
        enable_l0: true,
        l0_max_entries: 100,
        enable_l1: false, // Keep L1 off to isolate L0 behavior
        l1_max_memory: 0,
    };
    let cached = CachedTokenizer::new(inner.clone(), config);

    let input = "Hello";

    // 3. First call: add_special_tokens = true
    // This populates the L0 cache for the key "Hello"
    let enc1 = cached.encode(input, true).expect("First encode failed");
    let tokens_with_special = enc1.token_ids().to_vec();

    // 4. Second call: add_special_tokens = false
    // If the bug exists, the cache will hit on "Hello" and return the result
    // from step 3 (containing special tokens), even though we asked for none.
    let enc2 = cached.encode(input, false).expect("Second encode failed");
    let tokens_without_special = enc2.token_ids().to_vec();

    // 5. Verification
    println!("Tokens WITH special:    {:?}", tokens_with_special);
    println!("Tokens WITHOUT special: {:?}", tokens_without_special);

    // This assertion SHOULD pass in a correct implementation.
    // It will FAIL currently because both will be identical.
    assert_ne!(
        tokens_with_special,
        tokens_without_special,
        "BUG: L0 cache returned the same result for different add_special_tokens values! \
         This means the cache is returning results with special tokens when they were requested to be omitted."
    );
}
