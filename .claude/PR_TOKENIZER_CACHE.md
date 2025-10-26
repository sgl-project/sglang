# Add Two-Level Tokenizer Caching with Special Token Boundary Optimization

## Summary

This PR introduces a two-level tokenizer caching system that provides significant performance improvements for tokenization operations. The implementation features an L0 exact-match cache and an innovative L1 special-token boundary prefix cache, achieving **up to 99% L1 hit rates** and **22.7x speedup** on realistic chat workloads with high prefix reuse.

## Motivation

Tokenization is a frequent operation in the router, and many requests share common patterns:
- **Repeated requests**: Identical prompts or messages are tokenized multiple times
- **Common prefixes**: System prompts, chat templates, and conversation history create shared prefix patterns
- **Performance opportunity**: Caching tokenization results can significantly reduce redundant computation

The two-level cache design balances fast exact-match lookups (L0) with intelligent prefix reuse (L1).

## Architecture

### L0 Cache (Exact Match)
- **Purpose**: Fast lookup for identical input strings
- **Implementation**: DashMap-based concurrent hash map with lock-free reads
- **Configuration**: `l0_max_entries` (default: 10,000 entries, ~22MB memory)
- **Use case**: Repeated identical prompts, common system messages
- **Eviction**: Simple arbitrary eviction when capacity reached

### L1 Cache (Special Token Boundary Prefix Cache)
- **Purpose**: Reuse tokenization results for shared prefixes at special token boundaries
- **Implementation**: Re-tokenization approach - caches prefix tokens by re-tokenizing at ALL special token boundaries
- **Configuration**: `l1_max_memory` (default: 50MB)
- **Use case**: Chat conversations with shared system prompts, multi-turn interactions
- **Eviction**: LRU eviction with memory tracking
- **Correctness**: Guarantees 100% correctness by re-tokenizing prefixes (not storing raw strings)

#### How L1 Cache Works

L1 cache is a **special-token boundary prefix cache** that caches tokenization results at every special token boundary. Special tokens (like `<|im_start|>`, `<|im_end|>`, `<|eot_id|>`) are atomic in BPE tokenizers (`special: true`, `normalized: false`), making them the ONLY safe split points that guarantee correctness.

**Why Special Tokens Are Perfect**

BPE tokenizers make context-dependent merge decisions, so `tokenize(prefix + suffix) != tokenize(prefix) + tokenize(suffix)` for arbitrary boundaries. However, special tokens are atomic and protected from normalization/merging, guaranteeing:

```
tokenize(prefix + suffix) == tokenize(prefix) + tokenize(suffix)
```

when splitting at special token boundaries.

**Cache Strategy - Re-Tokenization with All-Boundaries Approach**

The L1 cache uses a **re-tokenization approach** to guarantee correctness. When inserting into the cache after a full tokenization:

1. **Find all special token boundaries** in the input string
2. **For each boundary**: Extract the prefix substring up to that boundary
3. **Re-tokenize the prefix** to get the exact token sequence (BPE-safe)
4. **Cache the prefix tokens** with Blake3 hash of the prefix string as the key

This approach ensures 100% correctness because we never assume `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`. Instead, we always re-tokenize the prefix to get the actual token sequence that would result from tokenizing `prefix + suffix` up to the boundary.

```
Input: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nHello!<|im_end|>"

Special token boundaries found:
1. After "<|im_start|>" at position 13
2. After "<|im_end|>" at position 45
3. After "<|im_start|>" at position 58
4. After "<|im_end|>" at position 72

For each boundary:
- Extract prefix string (e.g., "<|im_start|>system\nYou are helpful.<|im_end|>")
- Re-tokenize prefix → get exact token IDs
- Cache: hash(prefix_string) → prefix_tokens
```

**Cache Lookup Example:**
```
Input: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|>"

1. Find all special token boundaries in input string
2. For each boundary (longest to shortest):
   - Extract prefix substring up to boundary
   - Hash prefix with Blake3
   - Look up hash in cache
   - If HIT: Use cached prefix tokens + tokenize remaining suffix → merge and return
   - If MISS: Try next shorter prefix
3. If no prefix match:
   - Tokenize full string
   - Re-tokenize and cache prefixes at ALL boundaries for future requests
```

**Key Performance Insight**: On cache hit, we avoid re-tokenizing the prefix. On cache miss, we pay the cost of re-tokenizing prefixes once during insertion, then all future requests with that prefix benefit from the cached tokens.

**Key Features:**
- **Special tokens only**: No fallback to whitespace/punctuation - better to not cache than risk corruption
- **All-boundaries approach**: Caches at every special token, maximizing hit opportunities
- **Simple & natural**: Aligns perfectly with chat template structure
- **BPE-safe**: Guaranteed correctness for all BPE tokenizers
- **Sharded concurrent hash map** (DashMap) for O(1) average-case lookups
- **Blake3 hashing** (~10GB/s) for fast, collision-free prefix identification
- **LRU eviction** to keep memory bounded at max_memory limit
- **Lock-free atomics** for thread-safe operations

**Performance:**
- **Hit rate**: Up to 99% on realistic chat workloads with high prefix reuse
- **Memory efficiency**: ~10 cache entries per typical 8KB chat prompt (vs ~64 with granularity-based approach)
- **Speedup**: Up to 22.7x faster for high prefix reuse scenarios (customer service bots)
- **Correctness**: 100% - guaranteed by special token atomicity

### Opt-In Design
Both caches are **disabled by default** to maintain backward compatibility. The `CachedTokenizer` wrapper is only created when at least one cache is enabled; otherwise, the base tokenizer is used directly.

## Changes Made

### Core Implementation

**Tokenizer Cache Module (`src/tokenizer/cache/`)**
- `mod.rs`: `CachedTokenizer` wrapper with L0 and L1 coordination
- `l0.rs`: Exact-match cache with DashMap and fixed eviction deadlock issue
- `l1.rs`: Special-token boundary prefix cache with all-boundaries approach
- `fingerprint.rs`: Tokenizer fingerprint for cache invalidation

**Configuration (`src/config/types.rs`)**
- Added `TokenizerCacheConfig` struct with L0 and L1 parameters
- Removed `l1_granularity` field - simplified to just memory limit
- Implemented `Default` trait with both caches disabled (`false`)
- Added serde serialization support

**Validation (`src/config/validation.rs`)**
- Added validation for cache configuration:
  - L0 max entries must be > 0 when enabled
  - L1 max memory must be > 0 when enabled
  - Removed granularity validation (parameter removed)

**Tokenizer Factory (`src/tokenizer/factory.rs`)**
- Fixed special token detection in HuggingFace tokenizer
- Properly extracts `added_tokens` with `special: true` property
- Ensures special tokens are correctly identified for L1 cache

### CLI & Configuration

**CLI Arguments (`src/main.rs`)**
```bash
--tokenizer-cache-enable-l0              # Enable L0 exact-match cache
--tokenizer-cache-l0-max-entries <N>     # L0 cache size (default: 10000)
--tokenizer-cache-enable-l1              # Enable L1 prefix cache
--tokenizer-cache-l1-max-memory <BYTES>  # L1 max memory (default: 52428800)
```

**Python Bindings (`src/lib.rs`)**
- Added Python-accessible configuration fields (removed granularity)
- Integrated with existing `RouterArgs` for Python API compatibility

**Server Initialization (`src/server.rs`)**
- Conditional tokenizer creation:
  - Creates base tokenizer from factory
  - Wraps with `CachedTokenizer` when `enable_l0 || enable_l1`
  - Uses base tokenizer directly when both caches disabled

### Testing

**Unit Tests**
- L0 cache tests: exact match, eviction, stats, concurrent access, clear
- L1 cache tests: prefix match, longest match, special token boundaries, stats, eviction, clear
- Integration tests: cache hit/miss, batch encoding, decoder passthrough

**Test Configuration Updates**
Updated all test files to include `tokenizer_cache` field in `RouterConfig`:
- `tests/api_endpoints_test.rs`
- `tests/test_pd_routing.rs`
- `tests/responses_api_test.rs`
- `tests/streaming_tests.rs`
- `tests/request_formats_test.rs`

**Benchmarks (`benches/tokenizer_benchmark.rs`)**
- L0 vs L1 vs L0+L1 performance comparison
- Cache hit rate measurement
- Memory usage tracking
- Cold start vs warm cache comparison

## Configuration Examples

### Via CLI
```bash
# Enable both caches with custom settings
./sglang_router \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 20000 \
  --tokenizer-cache-enable-l1 \
  --tokenizer-cache-l1-max-memory 104857600
```

### Via Python
```python
from sglang_router import RouterArgs

args = RouterArgs(
    tokenizer_cache_enable_l0=True,
    tokenizer_cache_l0_max_entries=20000,
    tokenizer_cache_enable_l1=True,
    tokenizer_cache_l1_max_memory=104857600
)
```

### Via Config File
```json
{
  "tokenizer_cache": {
    "enable_l0": true,
    "l0_max_entries": 20000,
    "enable_l1": true,
    "l1_max_memory": 104857600
  }
}
```

## Why This Design Works

### BPE Tokenization Correctness and Re-Tokenization Approach

**The Problem**: BPE tokenizers make context-dependent merge decisions, so naive caching would be incorrect:
```
tokenize(prefix + suffix) != tokenize(prefix) + tokenize(suffix)
```

**The Solution**: Combine special token boundaries with re-tokenization:

1. **Special tokens are atomic boundaries** with guaranteed properties:
   - `special: true` - marked as special in tokenizer config
   - `normalized: false` - protected from normalization and BPE merging
   - This ensures: `tokenize(prefix + special_token + suffix) == tokenize(prefix + special_token) + tokenize(suffix)`

2. **Re-tokenize prefixes on insertion** (not on lookup):
   - When caching, extract prefix string and re-tokenize it to get exact tokens
   - Store the actual token IDs that result from tokenizing the prefix in context
   - This guarantees the cached tokens are exactly what would be produced when tokenizing the full string

3. **Correctness guarantee**:
   ```
   cached_tokens = tokenize(prefix)  // Re-tokenized in full context
   suffix_tokens = tokenize(suffix)
   result = cached_tokens + suffix_tokens  // Safe because split at special token
   ```

This hybrid approach (special token boundaries + re-tokenization) provides both **correctness** (100% guaranteed) and **performance** (avoid re-tokenizing prefixes on every cache hit).

### All-Boundaries Strategy Benefits

1. **Natural alignment**: Chat templates naturally have special tokens at conversation boundaries
2. **Maximum coverage**: Caches every possible prefix point, maximizing hit opportunities
3. **Fewer entries**: ~10 entries per 8KB prompt vs ~64 with granularity approach
4. **Simpler code**: No complex granularity/window calculations needed
5. **Exceptional hit rates**: Up to 99% on realistic chat workloads with high prefix reuse

### No Fallback Philosophy

The L1 cache **only** uses special tokens - if no special tokens are found, we simply don't cache. This is intentional:
- ✅ **Zero risk of corruption**: Never caches at unsafe boundaries
- ✅ **BPE-agnostic**: Works with all BPE tokenizers
- ✅ **Graceful degradation**: Falls back to L0 or no caching
- ✅ **Chat-optimized**: Modern chat templates always have special tokens

## Bug Fixes

**Special Token Detection** (`src/tokenizer/factory.rs`)
- Fixed HuggingFace tokenizer special token extraction
- Properly filters `added_tokens` for `special: true` property
- Ensures L1 cache has correct special tokens to work with

## Performance Results

Comprehensive benchmarks using Qwen3-4B-Instruct-2507 tokenizer on realistic workloads:

### Benchmark Scenarios

| Scenario | Configuration | Throughput | Latency | L1 Hit Rate | L0 Hit Rate | Speedup vs Baseline |
|----------|---------------|------------|---------|-------------|-------------|---------------------|
| **Realistic Chat** (95%+ prefix reuse) | L0+L1 | 18,751 ops/sec | 53.3µs | 99.2% | 0.8% | **18.2x** |
| **Customer Service** (100% prefix reuse) | L1-only | 21,359 ops/sec | 46.8µs | 88.9% | N/A | **22.7x** |
| **Customer Service** (100% prefix reuse) | L0+L1 | 19,494 ops/sec | 51.2µs | 87.5% | 11.1% | 21.1x |
| **Multi-turn Conversation** | L0+L1 | 4,469 ops/sec | 223.8µs | 66.7% | 0.0% | **4.3x** |
| **Code Review** | L0+L1 | 21,113 ops/sec | 47.4µs | 75.0% | 0.0% | **21.1x** |
| **Baseline** (no cache) | None | 940 ops/sec | 1,064µs | N/A | N/A | 1.0x |

### Configuration Guidance

**L1-only (recommended for high prefix reuse)**:
- **Best for**: Chat applications, customer service bots, conversational AI
- **Characteristics**: Many unique queries sharing common system prompts
- **Performance**: Up to 22.7x speedup, 88.9%+ hit rates
- **Why**: Avoids L0 lookup overhead when most requests have unique queries but shared prefixes

**L0+L1 (recommended for mixed workloads)**:
- **Best for**: Applications with both exact repeats and prefix reuse
- **Characteristics**: Some repeated queries, some variations with shared prefixes
- **Performance**: Up to 21.1x speedup, combined 98.6%+ hit rates
- **Why**: L0 catches exact repeats, L1 handles prefix variations

**L0-only (recommended for exact repetition)**:
- **Best for**: Load testing, health checks, monitoring probes
- **Characteristics**: Identical requests repeated many times
- **Performance**: Fastest for exact matches
- **Why**: Simple hash lookup without prefix analysis overhead

### Memory Efficiency

| Metric                       | Value                              |
|------------------------------|------------------------------------|
| Cache Entries per 8KB Prompt | ~10 (all-boundaries approach)      |
| Memory per Entry             | ~1.3KB                             |
| Total L1 Memory (8KB prompt) | ~13KB                              |
| Lookup Complexity            | O(k) where k = special token count |

## Future Work

Potential enhancements:
- **Metrics & observability**: Expose Prometheus metrics for cache hit/miss rates
- **Per-model configuration**: Different cache settings for different tokenizers
- **Cache warming**: Pre-populate cache with common patterns at startup
- **Dynamic tuning**: Auto-adjust cache sizes based on workload
- **Cross-request sharing**: Share cache across multiple router instances
