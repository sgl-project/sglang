//! Cache correctness integration test
//!
//! This test validates that the tokenizer cache (L0, L1, and L0+L1 combined) produces
//! exactly the same token IDs as uncached tokenization across multiple chat turns.
//! Uses the real Qwen/Qwen3-4B-Instruct-2507 tokenizer to test with actual special tokens.

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use sglang_router_rs::tokenizer::{
    cache::{CacheConfig, CachedTokenizer},
    hub::download_tokenizer_from_hf,
    huggingface::HuggingFaceTokenizer,
    traits::Encoder,
};

/// Global tokenizer path cache - download once, reuse across all tests
static TOKENIZER_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Download Qwen3-4B-Instruct-2507 tokenizer once and cache the path
async fn get_tokenizer_path() -> Option<PathBuf> {
    // Check if already downloaded
    if let Some(cached) = TOKENIZER_PATH.get() {
        return cached.clone();
    }

    // Download tokenizer
    let result = match download_tokenizer_from_hf("Qwen/Qwen3-4B-Instruct-2507").await {
        Ok(cache_dir) => {
            let tokenizer_path = cache_dir.join("tokenizer.json");
            if tokenizer_path.exists() {
                Some(tokenizer_path)
            } else {
                println!("Tokenizer downloaded but tokenizer.json not found");
                None
            }
        }
        Err(e) => {
            println!("Failed to download tokenizer: {}", e);
            None
        }
    };

    // Cache the result (even if None, so we don't retry on failure)
    TOKENIZER_PATH.set(result.clone()).ok();
    result
}

/// Comprehensive multi-turn chat conversation for testing cache correctness
/// Uses Qwen's special tokens with diverse content to hit edge cases
const CHAT_TURNS: [&str; 29] = [
    // Basic conversation patterns
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>",
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|><|im_start|>user\nWhat is the capital of France?<|im_end|>",
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|><|im_start|>user\nWhat is the capital of France?<|im_end|><|im_start|>assistant\nThe capital of France is Paris.<|im_end|>",

    // Different system prompts (testing different prefix patterns)
    "<|im_start|>system\nYou are a coding tutor specializing in Rust programming.<|im_end|><|im_start|>user\nExplain ownership.<|im_end|>",
    "<|im_start|>system\nYou are a math teacher.<|im_end|><|im_start|>user\nSolve: 2x + 5 = 13<|im_end|>",

    // Long conversation with multiple turns (testing longer prefixes)
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|><|im_start|>user\nTell me about deep learning.<|im_end|><|im_start|>assistant\nDeep learning is a subset of machine learning that uses neural networks with multiple layers.<|im_end|><|im_start|>user\nWhat are the main architectures?<|im_end|>",

    // Code snippets (testing different character patterns)
    "<|im_start|>system\nYou are a code reviewer.<|im_end|><|im_start|>user\nReview this code:\nfn main() {\n    println!(\"Hello, world!\");\n}\n<|im_end|>",
    "<|im_start|>system\nYou are a code reviewer.<|im_end|><|im_start|>user\nExplain this Rust code:\nimpl<T> Drop for Box<T> {\n    fn drop(&mut self) { /* ... */ }\n}\n<|im_end|>",

    // Mathematical content
    "<|im_start|>system\nYou are a math tutor.<|im_end|><|im_start|>user\nProve that √2 is irrational using proof by contradiction.<|im_end|>",
    "<|im_start|>system\nYou are a math tutor.<|im_end|><|im_start|>user\nCalculate: ∫(x² + 3x + 2)dx from 0 to 5<|im_end|>",

    // Multilingual content
    "<|im_start|>system\nYou are a multilingual assistant.<|im_end|><|im_start|>user\nTranslate to French: The quick brown fox jumps over the lazy dog.<|im_end|>",
    "<|im_start|>system\nYou are a multilingual assistant.<|im_end|><|im_start|>user\n你好，请帮我翻译这句话：I love programming in Rust.<|im_end|>",
    "<|im_start|>system\nYou are a multilingual assistant.<|im_end|><|im_start|>user\nこんにちは！Rustについて教えてください。<|im_end|>",

    // Special characters and emojis
    "<|im_start|>system\nYou are a friendly chatbot.<|im_end|><|im_start|>user\nWhat do you think about emojis? 😀🎉🚀💻<|im_end|>",
    "<|im_start|>system\nYou are a data analyst.<|im_end|><|im_start|>user\nAnalyze this: {\"name\": \"test\", \"value\": 42, \"nested\": {\"key\": \"value\"}}<|im_end|>",

    // Very long message (testing large token counts)
    "<|im_start|>system\nYou are a literature expert.<|im_end|><|im_start|>user\nAnalyze the themes in this passage: In the vast expanse of the digital realm, where bits and bytes dance in harmonious symphony, there exists a paradigm that transcends mere computation. This paradigm, known as machine learning, represents humanity's quest to imbue silicon with the spark of cognition. Deep neural networks, inspired by the intricate architecture of biological brains, layer upon layer of artificial neurons, each connection a synapse firing in the dark recesses of mathematical space. Through gradient descent, these networks learn patterns invisible to human perception, extracting meaning from chaos, signal from noise. The transformer architecture revolutionized this field, introducing attention mechanisms that allowed models to focus on relevant information, much like how humans selectively attend to important details in their environment.<|im_end|>",

    // Edge case: Multiple special tokens in sequence
    "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello!<|im_end|><|im_start|>user\nHow are you?<|im_end|>",

    // Edge case: Empty-ish messages
    "<|im_start|>system\n<|im_end|><|im_start|>user\nTest<|im_end|>",
    "<|im_start|>system\nBrief.<|im_end|><|im_start|>user\nOK<|im_end|>",

    // Technical documentation style
    "<|im_start|>system\nYou are a technical writer.<|im_end|><|im_start|>user\nDocument the following API:\n\n```rust\npub struct CachedTokenizer {\n    inner: Arc<dyn Tokenizer>,\n    l0: Option<L0Cache>,\n    l1: Option<L1Cache>,\n}\n\nimpl Encoder for CachedTokenizer {\n    fn encode(&self, input: &str) -> Result<Encoding>;\n}\n```\n<|im_end|>",

    // Conversation with code review
    "<|im_start|>system\nYou are a senior Rust developer.<|im_end|><|im_start|>user\nReview for correctness:\n\nlet special_tokens: Option<Vec<&str>> = self.l1.as_ref().map(|_| {\n    self.special_token_strings.iter().map(|s| s.as_str()).collect()\n});<|im_end|>",

    // Markdown formatted content
    "<|im_start|>system\nYou are a documentation assistant.<|im_end|><|im_start|>user\nFormat this as markdown:\n\n# Cache Architecture\n\n## L0 Cache\n- Exact match\n- DashMap based\n- 10K entries\n\n## L1 Cache  \n- Prefix match\n- Special token boundaries\n- 50MB memory\n<|im_end|>",

    // Complex nested structures
    "<|im_start|>system\nYou are a JSON expert.<|im_end|><|im_start|>user\nValidate this JSON:\n{\n  \"tokenizer_cache\": {\n    \"enable_l0\": true,\n    \"l0_max_entries\": 10000,\n    \"enable_l1\": true,\n    \"l1_max_memory\": 52428800,\n    \"stats\": {\n      \"hits\": [1, 2, 3],\n      \"misses\": {\"count\": 5}\n    }\n  }\n}\n<|im_end|>",

    // SQL queries
    "<|im_start|>system\nYou are a database expert.<|im_end|><|im_start|>user\nOptimize this query:\nSELECT u.name, COUNT(p.id) as post_count\nFROM users u\nLEFT JOIN posts p ON u.id = p.user_id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id, u.name\nHAVING COUNT(p.id) > 5\nORDER BY post_count DESC;<|im_end|>",

    // Regex patterns
    "<|im_start|>system\nYou are a regex expert.<|im_end|><|im_start|>user\nExplain this regex: ^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\\.)+[a-zA-Z]{2,}$<|im_end|>",

    // Command line examples
    "<|im_start|>system\nYou are a DevOps engineer.<|im_end|><|im_start|>user\nExplain this command:\ncargo bench --bench tokenizer_benchmark -- --color=never | tee results.txt<|im_end|>",

    // Unicode edge cases
    "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nTest: café, naïve, Zürich, 北京, 東京, मुंबई, Москва<|im_end|>",

    // Mixed content complexity
    "<|im_start|>system\nYou are a software architect.<|im_end|><|im_start|>user\nDesign a caching system that:\n1. Handles 10K+ QPS\n2. Maintains 99.9% uptime  \n3. Supports L0 (exact) and L1 (prefix) caching\n4. Uses Blake3 for hashing (10GB/s throughput)\n5. Implements LRU eviction\n6. Thread-safe with lock-free reads\n\nKey requirements:\n- Memory: 50MB L1 budget\n- Latency: <100µs p99\n- Correctness: 100% (no false tokens)\n<|im_end|>",

    // Very long technical discussion
    "<|im_start|>system\nYou are a compiler expert.<|im_end|><|im_start|>user\nExplain why BPE tokenizers are not prefix-stable:\n\nThe core issue is that BPE applies merges based on local context. When you tokenize 'prefix' alone, it might apply merge rules differently than when tokenizing 'prefix + suffix' as a whole. For example:\n\ntokenize('hello world') might produce [hello, _world]\ntokenize('hello') + tokenize(' world') might produce [hel, lo, _wo, rld]\n\nThis is because the merge rules see different contexts. The space before 'world' in the first case is part of the token boundary, but in the second case, ' world' is tokenized in isolation.\n\nSpecial tokens solve this because they are:\n1. Atomic (never split or merged)\n2. Protected from normalization\n3. Marked with special: true flag\n4. Have normalized: false property\n\nThis guarantees: tokenize(prefix + special + suffix) = tokenize(prefix + special) + tokenize(suffix)\n\nOur L1 cache exploits this by:\n1. Finding all special token boundaries\n2. Re-tokenizing prefixes at those boundaries\n3. Caching the exact token IDs\n4. On cache hit, appending suffix tokens\n\nThis achieves both correctness (100%) and performance (22.7x speedup on high prefix reuse workloads).<|im_end|>",
];

#[tokio::test]
async fn test_cache_produces_identical_tokens() {
    // Get tokenizer path (download once, cached across tests)
    let tokenizer_path = match get_tokenizer_path().await {
        Some(path) => path,
        None => {
            println!("Skipping test - tokenizer not available");
            return;
        }
    };

    // Create base tokenizer (no cache)
    let base_tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load base tokenizer"),
    );

    // Create cached tokenizers with different configurations
    let l0_only_config = CacheConfig {
        enable_l0: true,
        l0_max_entries: 10_000,
        enable_l1: false,
        l1_max_memory: 0,
    };

    let l1_only_config = CacheConfig {
        enable_l0: false,
        l0_max_entries: 0,
        enable_l1: true,
        l1_max_memory: 50 * 1024 * 1024,
    };

    let l0_l1_config = CacheConfig {
        enable_l0: true,
        l0_max_entries: 10_000,
        enable_l1: true,
        l1_max_memory: 50 * 1024 * 1024,
    };

    let l0_tokenizer = Arc::new(CachedTokenizer::new(base_tokenizer.clone(), l0_only_config));
    let l1_tokenizer = Arc::new(CachedTokenizer::new(base_tokenizer.clone(), l1_only_config));
    let l0_l1_tokenizer = Arc::new(CachedTokenizer::new(base_tokenizer.clone(), l0_l1_config));

    println!(
        "\n=== Testing Cache Correctness Across {} Chat Turns ===\n",
        CHAT_TURNS.len()
    );

    for (turn_idx, turn) in CHAT_TURNS.iter().enumerate() {
        println!("Turn {}: Testing {} chars", turn_idx + 1, turn.len());

        // Tokenize with base (no cache)
        let base_encoding = base_tokenizer
            .encode(turn)
            .expect("Base tokenization failed");
        let base_tokens = base_encoding.token_ids();

        // Tokenize with L0-only
        let l0_encoding = l0_tokenizer.encode(turn).expect("L0 tokenization failed");
        let l0_tokens = l0_encoding.token_ids();

        // Tokenize with L1-only
        let l1_encoding = l1_tokenizer.encode(turn).expect("L1 tokenization failed");
        let l1_tokens = l1_encoding.token_ids();

        // Tokenize with L0+L1
        let l0_l1_encoding = l0_l1_tokenizer
            .encode(turn)
            .expect("L0+L1 tokenization failed");
        let l0_l1_tokens = l0_l1_encoding.token_ids();

        // Verify all configurations produce identical token IDs
        assert_eq!(
            base_tokens.len(),
            l0_tokens.len(),
            "Turn {}: L0 token count mismatch (base: {}, L0: {})",
            turn_idx + 1,
            base_tokens.len(),
            l0_tokens.len()
        );

        assert_eq!(
            base_tokens.len(),
            l1_tokens.len(),
            "Turn {}: L1 token count mismatch (base: {}, L1: {})",
            turn_idx + 1,
            base_tokens.len(),
            l1_tokens.len()
        );

        assert_eq!(
            base_tokens.len(),
            l0_l1_tokens.len(),
            "Turn {}: L0+L1 token count mismatch (base: {}, L0+L1: {})",
            turn_idx + 1,
            base_tokens.len(),
            l0_l1_tokens.len()
        );

        // Compare token by token
        for (token_idx, (((base_token, l0_token), l1_token), l0_l1_token)) in base_tokens
            .iter()
            .zip(l0_tokens.iter())
            .zip(l1_tokens.iter())
            .zip(l0_l1_tokens.iter())
            .enumerate()
        {
            assert_eq!(
                base_token,
                l0_token,
                "Turn {}, token {}: L0 mismatch (base: {}, L0: {})",
                turn_idx + 1,
                token_idx,
                base_token,
                l0_token
            );

            assert_eq!(
                base_token,
                l1_token,
                "Turn {}, token {}: L1 mismatch (base: {}, L1: {})",
                turn_idx + 1,
                token_idx,
                base_token,
                l1_token
            );

            assert_eq!(
                base_token,
                l0_l1_token,
                "Turn {}, token {}: L0+L1 mismatch (base: {}, L0+L1: {})",
                turn_idx + 1,
                token_idx,
                base_token,
                l0_l1_token
            );
        }

        println!(
            "  ✓ All configurations produced identical {} tokens",
            base_tokens.len()
        );
    }

    // Print cache statistics
    if let Some(l0_stats) = l0_tokenizer.cache_stats() {
        println!("\n=== L0 Cache Statistics ===");
        println!("  Hits: {}", l0_stats.hits);
        println!("  Misses: {}", l0_stats.misses);
        println!(
            "  Hit rate: {:.2}%",
            if l0_stats.hits + l0_stats.misses > 0 {
                l0_stats.hits as f64 / (l0_stats.hits + l0_stats.misses) as f64 * 100.0
            } else {
                0.0
            }
        );
        println!("  Entries: {}", l0_stats.entries);
    }

    if let Some(l1_stats) = l1_tokenizer.l1_cache_stats() {
        println!("\n=== L1 Cache Statistics ===");
        println!("  Hits: {}", l1_stats.hits);
        println!("  Misses: {}", l1_stats.misses);
        println!(
            "  Hit rate: {:.2}%",
            if l1_stats.hits + l1_stats.misses > 0 {
                l1_stats.hits as f64 / (l1_stats.hits + l1_stats.misses) as f64 * 100.0
            } else {
                0.0
            }
        );
        println!("  Entries: {}", l1_stats.entries);
        println!("  Memory used: {} bytes", l1_stats.memory_bytes);
    }

    if let Some(l0_stats) = l0_l1_tokenizer.cache_stats() {
        if let Some(l1_stats) = l0_l1_tokenizer.l1_cache_stats() {
            println!("\n=== L0+L1 Combined Cache Statistics ===");
            println!("  L0 Hits: {}", l0_stats.hits);
            println!("  L1 Hits: {}", l1_stats.hits);
            println!(
                "  Total Hit rate: {:.2}%",
                if l0_stats.hits + l1_stats.hits + l0_stats.misses + l1_stats.misses > 0 {
                    (l0_stats.hits + l1_stats.hits) as f64
                        / (l0_stats.hits + l1_stats.hits + l0_stats.misses + l1_stats.misses) as f64
                        * 100.0
                } else {
                    0.0
                }
            );
        }
    }

    println!("\n✓ All cache configurations produce identical tokenization results!");
}

#[tokio::test]
async fn test_cache_correctness_with_edge_cases() {
    // Get tokenizer path (download once, cached across tests)
    let tokenizer_path = match get_tokenizer_path().await {
        Some(path) => path,
        None => {
            println!("Skipping test - tokenizer not available");
            return;
        }
    };

    // Create base and cached tokenizers
    let base_tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load base tokenizer"),
    );

    let cached_config = CacheConfig {
        enable_l0: true,
        l0_max_entries: 10_000,
        enable_l1: true,
        l1_max_memory: 50 * 1024 * 1024,
    };

    let cached_tokenizer = Arc::new(CachedTokenizer::new(base_tokenizer.clone(), cached_config));

    println!("\n=== Testing Edge Cases and Complex Patterns ===\n");

    // Edge cases that stress-test the cache
    let edge_cases = [
        // Minimal messages
        ("<|im_start|>system\n<|im_end|>", "Empty system message"),
        ("<|im_start|>user\na<|im_end|>", "Single character"),

        // Special token boundaries
        ("<|im_start|>system\nA<|im_end|><|im_start|>user\nB<|im_end|><|im_start|>assistant\nC<|im_end|>", "Minimal multi-turn"),

        // Repeated exact queries (L0 hit test)
        ("<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nHello!<|im_end|>", "Repeated query 1"),
        ("<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nHello!<|im_end|>", "Repeated query 2"),

        // Same prefix, different suffix (L1 hit test)
        ("<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nWhat is 1+1?<|im_end|>", "Same prefix, diff suffix 1"),
        ("<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|>", "Same prefix, diff suffix 2"),
        ("<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nWhat is 3+3?<|im_end|>", "Same prefix, diff suffix 3"),

        // Unicode stress tests
        ("<|im_start|>system\n你好<|im_end|><|im_start|>user\n世界<|im_end|>", "Chinese characters"),
        ("<|im_start|>system\nこんにちは<|im_end|><|im_start|>user\n世界<|im_end|>", "Japanese + Chinese"),
        ("<|im_start|>system\n🚀💻🎉<|im_end|><|im_start|>user\n😀😃😄<|im_end|>", "Emoji only"),

        // Whitespace edge cases
        ("<|im_start|>system\n   \n<|im_end|>", "Whitespace only"),
        ("<|im_start|>system\n\n\n\n<|im_end|>", "Multiple newlines"),
        ("<|im_start|>system\n\t\t\t<|im_end|>", "Tabs"),

        // Long token sequences
        ("<|im_start|>system\nThe quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.<|im_end|>", "Repeated phrase"),

        // Special characters
        ("<|im_start|>system\n!@#$%^&*()_+-=[]{}|;':\",./<>?<|im_end|>", "ASCII special chars"),
        ("<|im_start|>system\n`~\\<|im_end|>", "Backtick and tilde"),

        // Code with special formatting
        ("<|im_start|>system\nCode: fn() -> Result<(), Box<dyn Error>><|im_end|>", "Rust generics"),
        ("<|im_start|>system\nRegex: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$<|im_end|>", "Email regex"),

        // Very long single token sequences (testing buffer handling)
        ("<|im_start|>system\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<|im_end|>", "Repeated 'a'"),
        ("<|im_start|>system\n0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789<|im_end|>", "Repeated numbers"),
    ];

    let mut test_count = 0;
    let mut mismatch_count = 0;

    for (query, description) in edge_cases.iter() {
        test_count += 1;

        let base_tokens = base_tokenizer
            .encode(query)
            .expect("Base encoding failed")
            .token_ids()
            .to_vec();

        let cached_tokens = cached_tokenizer
            .encode(query)
            .expect("Cached encoding failed")
            .token_ids()
            .to_vec();

        if base_tokens != cached_tokens {
            mismatch_count += 1;
            println!("  ✗ {}: Token mismatch!", description);
            println!(
                "    Base length: {}, Cached length: {}",
                base_tokens.len(),
                cached_tokens.len()
            );

            // Show first few mismatching tokens for debugging
            for (i, (base, cached)) in base_tokens.iter().zip(cached_tokens.iter()).enumerate() {
                if base != cached {
                    println!("    Token {}: base={}, cached={}", i, base, cached);
                    if i >= 5 {
                        break;
                    }
                }
            }
        } else {
            println!("  ✓ {}: {} tokens", description, base_tokens.len());
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "{} out of {} edge cases failed!",
        mismatch_count, test_count
    );

    // Print cache statistics
    if let Some(l0_stats) = cached_tokenizer.cache_stats() {
        println!("\n=== Cache Statistics ===");
        println!(
            "  L0 Hits: {} ({:.1}% hit rate)",
            l0_stats.hits,
            if l0_stats.hits + l0_stats.misses > 0 {
                l0_stats.hits as f64 / (l0_stats.hits + l0_stats.misses) as f64 * 100.0
            } else {
                0.0
            }
        );
    }

    if let Some(l1_stats) = cached_tokenizer.l1_cache_stats() {
        println!(
            "  L1 Hits: {} ({:.1}% hit rate)",
            l1_stats.hits,
            if l1_stats.hits + l1_stats.misses > 0 {
                l1_stats.hits as f64 / (l1_stats.hits + l1_stats.misses) as f64 * 100.0
            } else {
                0.0
            }
        );
    }

    println!("\n✓ All {} edge cases passed!", test_count);
}
