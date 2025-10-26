# L1 Cache: Technical Deep Dive

## Overview

L1 cache is a **fixed-boundary prefix cache** that caches tokenization results at regular byte intervals (e.g., every 128 bytes). It's designed for chat templates where different requests share common prefixes (like system prompts).

---

## Architecture

### Core Data Structures

#### 1. **L1Cache Struct**
```rust
pub struct L1Cache {
    shards: Vec<Arc<DashMap<Blake3Hash, CachedPrefix>>>,  // 16 shards
    granularity: usize,                                    // 128 bytes
    max_memory: usize,                                     // 50MB default
    current_memory: AtomicU64,                             // Current usage
    hits: AtomicU64,                                       // Hit counter
    misses: AtomicU64,                                     // Miss counter
    access_counter: AtomicU64,                             // LRU timestamp
}
```

#### 2. **CachedPrefix Struct**
```rust
struct CachedPrefix {
    tokens: Vec<TokenIdType>,           // Cached tokens
    last_accessed: Arc<AtomicU64>,      // LRU timestamp
    size_bytes: usize,                  // Memory tracking
}
```

---

## Key Dependencies & Packages

### 1. **DashMap** (`dashmap = "6.1.0"`)
- **What**: Lock-free concurrent HashMap
- **Why**: Allows multiple threads to read/write cache simultaneously without locks
- **Implementation**: Uses internal sharding (we add 16 additional shards on top)
- **Performance**: O(1) average lookup/insert with minimal contention

### 2. **Blake3** (`blake3 = "1.5"`)
- **What**: Cryptographic hash function
- **Why**:
  - Extremely fast (faster than SipHash/FxHash for our use case)
  - No collisions for practical purposes (2^256 space)
  - Consistent across platforms
- **Output**: 32-byte hash `[u8; 32]`
- **Performance**: ~10GB/s on modern CPUs

### 3. **AtomicU64** (std::sync::atomic)
- **What**: Lock-free atomic 64-bit integer
- **Why**: Thread-safe counters without mutex overhead
- **Operations**: `fetch_add`, `load`, `store` with Ordering::Relaxed

---

## How It Works

### Cache Insertion Flow

```
Input: "System: You are helpful.\n\nUser: Hello"  (200 bytes)
Granularity: 128 bytes

Step 1: Calculate boundaries
  - bytes.len() = 200
  - Boundaries: 128 (only one, since 256 > 200)

Step 2: For each boundary (k = 128):
  a) Extract prefix: bytes[0..128]
  b) Hash prefix: Blake3(prefix) → [u8; 32]
  c) Shard selection: hash[0] % 16 → shard_idx
  d) Estimate tokens: (128/200) * total_tokens
  e) Store in DashMap: hash → CachedPrefix

Step 3: Update memory counter
  - size = 128 + (num_tokens * 4)  // 4 bytes per u32 token
  - current_memory += size
```

### Cache Lookup Flow

```
Input: "System: You are helpful.\n\nUser: How are you?"  (210 bytes)
Granularity: 128 bytes

Step 1: Calculate max_boundary
  - max_boundary = (210 / 128) * 128 = 128

Step 2: Search backwards from longest boundary
  - Try k = 128:
    a) prefix = bytes[0..128]
    b) hash = Blake3(prefix)
    c) shard_idx = hash[0] % 16
    d) Look up in DashMap: shards[shard_idx].get(hash)

    If found:
      - Update LRU timestamp
      - Return (cached_tokens, 128)
    If not found:
      - Continue to next boundary (none in this case)

Step 3: Return None (cache miss)
```

---

## Runtime Complexity

### Insertion: `O(n/g + m)`
- **n**: Input length in bytes
- **g**: Granularity (128 bytes)
- **m**: Number of entries to evict (typically 0)

**Breakdown**:
```
For each boundary k in [g, 2g, 3g, ..., n]:
  - Blake3 hash:        O(k) amortized → O(g) per boundary
  - DashMap insert:     O(1) expected
  - Token estimation:   O(1)

Total: O(n/g) for hashing + O(n/g) for insertion = O(n/g)

If eviction needed:
  - Collect entries:    O(total_entries)
  - Sort by timestamp:  O(m log m)
  - Remove m entries:   O(m)

Total with eviction: O(n/g + m log m)
```

**Example**: 8KB input with 128-byte granularity
- Boundaries: 8192/128 = 64
- Operations: 64 hashes + 64 inserts ≈ 128 operations
- Time: ~100-200 µs

### Lookup: `O(n/g)`

```
For each boundary k in [max_boundary, max_boundary-g, ..., g] (descending):
  - Blake3 hash:        O(k) → varies by boundary
  - DashMap get:        O(1) expected

Worst case: Check all boundaries = O(n/g) hashes
Best case: Hit on first try = O(n/g) for largest hash
Average: Hit halfway = O(n/(2g)) hashes
```

**Example**: 8KB input
- Max boundary: 8192 bytes
- First hash: Blake3(8192 bytes) ≈ 1 µs
- DashMap lookup: ~10 ns
- **Total if hit**: ~1 µs

### LRU Eviction: `O(E log E)` where E = total entries

```
1. Collect all entries: O(E)
   - Iterate 16 shards
   - For each entry, read timestamp

2. Sort by timestamp: O(E log E)
   - Rust's sort is TimSort (stable)

3. Remove entries: O(m) where m = entries to evict
   - DashMap remove: O(1) per entry
```

**Example**: 1000 entries, need to evict 100
- Collect: 1000 iterations ≈ 10 µs
- Sort: 1000 * log(1000) ≈ 10,000 ops ≈ 50 µs
- Remove: 100 removes ≈ 10 µs
- **Total**: ~70 µs

---

## Memory Layout

### Per Cache Entry
```
CachedPrefix {
    tokens: Vec<u32>,              // 24 bytes (Vec header) + 4*n bytes
    last_accessed: Arc<AtomicU64>, // 16 bytes (Arc + AtomicU64)
    size_bytes: usize,             // 8 bytes
}

Total overhead: ~48 bytes + token data
```

### Hash Map Storage
```
DashMap<[u8; 32], CachedPrefix>

Key:   32 bytes (Blake3 hash)
Value: ~48 bytes + token data
Total per entry: ~80 bytes + token data
```

### Example Memory Usage
```
Input: 8KB system prompt with 2000 tokens
Granularity: 128 bytes
Boundaries: 8192/128 = 64 entries

Per entry (average):
  - Tokens: (2000/64) ≈ 31 tokens * 4 = 124 bytes
  - Overhead: 80 bytes
  - Total: 204 bytes

Total cache size: 64 * 204 ≈ 13KB for one prompt
```

---

## Concurrency Model

### Thread Safety

1. **DashMap**: Lock-free reads, fine-grained locking on writes
2. **AtomicU64**: Lock-free counters with relaxed ordering
3. **Arc**: Reference counting for shared ownership

### Sharding Strategy

```
16 shards determined by: hash[0] % 16

Why 16?
  - Balances contention vs overhead
  - 16 threads can operate independently
  - Fits in CPU cache lines well
  - Power of 2 for fast modulo

Shard distribution:
  - Blake3 provides uniform distribution
  - Each shard gets ~1/16 of traffic
```

### Lock-Free Operations

```rust
// All these are lock-free:
access_counter.fetch_add(1, Ordering::Relaxed)
hits.fetch_add(1, Ordering::Relaxed)
current_memory.fetch_add(size, Ordering::Relaxed)
last_accessed.store(timestamp, Ordering::Relaxed)

// DashMap operations (internally synchronized):
shards[idx].get(&hash)      // Read lock (optimized)
shards[idx].insert(hash, v) // Write lock (brief)
```

---

## Production Characteristics

### Strengths ✅

1. **Scalable**: O(1) average lookup with 16-way sharding
2. **Thread-safe**: Lock-free counters, fine-grained DashMap locks
3. **Memory efficient**: LRU eviction keeps memory bounded
4. **Fast hashing**: Blake3 at ~10GB/s
5. **Predictable**: Fixed boundaries = deterministic cache behavior

### Weaknesses ⚠️

1. **Eviction cost**: O(E log E) when cache is full
2. **Hash overhead**: Must hash large prefixes (mitigated by Blake3 speed)
3. **Boundary alignment**: Only caches at 128-byte boundaries
4. **No partial hits**: Either full boundary match or miss

### Optimization Opportunities 🚀

1. **Incremental hashing**: Cache intermediate Blake3 states
2. **Lazy eviction**: Evict in background thread
3. **Adaptive granularity**: Smaller boundaries for short prompts
4. **Bloom filter**: Pre-filter misses before hashing

---

## Comparison with Alternatives

| Approach | Lookup | Insert | Memory | Thread-Safe |
|----------|--------|--------|--------|-------------|
| **L1 (DashMap + Blake3)** | O(n/g) | O(n/g) | O(E*T) | ✅ Lock-free |
| Trie | O(n) | O(n) | O(n*E) | ❌ Needs locks |
| Suffix Array | O(n log E) | O(E log E) | O(E*T) | ❌ Needs locks |
| Rolling Hash | O(n) | O(n) | O(E*T) | ⚠️ Collision risk |

Where:
- **n**: Input length
- **g**: Granularity (128)
- **E**: Total entries
- **T**: Average tokens per entry

---

## Real-World Performance

### Benchmark Results (8KB prompts)

```
Uncached:        1359.4 µs  (baseline)
L1-only:          663.0 µs  (2.05x speedup)
L0+L1:            686.2 µs  (1.98x speedup)
L0+L1 (reuse):    234.0 µs  (5.81x speedup)
```

### Cache Efficiency

```
Hit Rate: 80%
  - System prompt (8KB): Cached at boundaries
  - User query: Full tokenization needed

Memory Usage: ~13KB per unique 8KB prompt
Cache Capacity: 50MB = ~3,800 unique prompts
```

### Production Recommendations

```yaml
Configuration:
  max_memory: 50MB      # ~4K unique prompts
  granularity: 128      # Good balance
  enable_l0: true       # For exact match reuse
  enable_l1: true       # For prefix reuse

Expected Hit Rates:
  - Chat templates: 70-85%
  - API calls: 50-70%
  - Batch jobs: 40-60%
```

---

## Summary

L1 cache uses a **sharded concurrent hash map** (DashMap) with **Blake3 hashing** for O(1) average-case lookups. It caches at **fixed 128-byte boundaries**, making it ideal for chat templates with shared system prompts. The **LRU eviction** keeps memory bounded, while **lock-free atomics** ensure thread safety. Runtime complexity is **O(n/g)** for both insertion and lookup, with **O(E log E)** eviction when needed.
