# RFC: Marconi Prefix Caching for Hybrid LLMs in SGLang

Related Paper: [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/pdf/2411.19379)

Related Code: https://github.com/ruipeterpan/marconi/tree/main

---

## 1. Summary

This RFC proposes integrating Marconi's prefix caching into SGLang. We extend SGLang's existing support for Hybrid Architecture with Marconi's FLOP-efficiency-weighted cache eviction policy.

Target Model: `Qwen/Qwen3.5-27B`

---

## 2. Existing Hybrid Model Support in SGLang

SGLang already has a complete prefix caching system for hybrid (attention + SSM) models.

### 2.1 TreeNode: Dual-State Radix Tree Node

Each `TreeNode` is shared across RadixTree and both LRU lists.

```python
class TreeNode:
    children: defaultdict[int, TreeNode]
    parent: TreeNode
    key: RadixKey

    # --- Cache state ---
    value: Optional[Tensor]       # indices into TokenToKVPool (attention KV cache)
    mamba_value: Optional[Tensor] # indices into MambaPool (SSM conv + temporal state)

    # --- Lock reference counts ---
    full_lock_ref: int   # KV lock
    mamba_lock_ref: int  # SSM lock

    # --- LRU linked-list pointers ---
    prev, next: TreeNode                  # for full_lru_list (KV eviction)
    mamba_prev, mamba_next: TreeNode      # for mamba_lru_list (SSM eviction)
    last_access_time: float64             # for LRU ordering
```

### 2.2 MambaRadixCache

`MambaRadixCache` holds a single radix tree of `TreeNode`s, two independent LRU lists, and references to the index allocators:

```python
class MambaRadixCache(BasePrefixCache):
    root_node: TreeNode                                    # root of single radix tree

    # Free list tracker to store which slots are in use vs available
    req_to_token_pool: HybridReqToTokenPool                # free pages management for SSM slots
    token_to_kv_pool_allocator: TokenToKVPoolAllocator     # free_pages management for KV slots

    # Two LRU lists for eviction ordering over the same set of nodes
    full_lru_list: LRUList    # LRU list for KV Cache, can evict leaf nodes only
    mamba_lru_list: LRUList   # LRU list for SSM state cache, can evict any node (including tombstoning)
```

The current eviction policy is pure LRU for both lists.
We will be replacing this eviction path with Marconi's FLOP-efficiency-weighted scoring.

### 2.3 GPU Memory Pools

**Pool 1 — KV Cache** (for full-attention layers):
```python
# Pre-allocated GPU tensors in MHATokenToKVPool
k_buffer = [torch.zeros(total_slots, head_num, head_dim) for _ in range(num_attn_layers)]
v_buffer = [torch.zeros(total_slots, head_num, head_dim) for _ in range(num_attn_layers)]

# Free-list in TokenToKVPoolAllocator
free_pages: Tensor[int64]
```

**Pool 2 — SSM State** (for linear-attention/Mamba layers):
```python
# Pre-allocated GPU tensors in MambaPool
conv_state     = torch.zeros(num_mamba_layers, pool_size, conv_dim/tp, conv_kernel-1)
temporal_state = torch.zeros(num_mamba_layers, pool_size, num_heads/tp, head_dim, state_size)

# Free-list also in MambaPool
free_slots: Tensor[int64]
```

### 2.4 KV and SSM State Eviction Invariant

- **SSM can be evicted without KV** (tombstoning)
- **KV cannot be evicted without also evicting SSM**

Whenever a node's KV cache is evicted, its SSM state must be freed simultaneously.

## 3. Scope

We will add a scoring layer into the eviction decision of `MambaRadixCache`.

### In Scope

- **Cache eviction policy**: Replace pure LRU with FLOP-efficiency-weighted scoring in `evict_mamba()` and `evict()`
- **FLOP utility functions**: New `marconi_utils.py` with FLOP/memory calculations for Qwen3-Next
- **Timestamp update**: Update only the terminal node on prefix match (not all ancestors)

### Out of Scope

- **Cache admission policy**: No changes to when SSM states are cached

## 4. API Summary
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --eviction-policy marconi \
    --marconi-eff-weight 0.7
```
### 5. Backward Compatibility

- Default `eviction_policy="lru"`
- No changes to `MambaRadixCache`'s public API (`match_prefix`, `insert`, `evict`, `cache_finished_req`, `cache_unfinished_req`)
- All existing hybrid models (Falcon-H1, Nemotron-H, LFM2, etc.) continue to use LRU

## 6. Changes Required

### 6.1 Add Server Args and CacheInitParams

**`server_args.py`** — new CLI arguments:
```python
parser.add_argument(
    "--eviction-policy",
    type=str, default="lru", choices=["lru", "marconi"],
    help="Cache eviction policy for hybrid models. "
         "'marconi' enables FLOP-aware eviction."
)
parser.add_argument(
    "--marconi-eff-weight",
    type=float, default=0.5,
    help="Weight for FLOP efficiency term in Marconi eviction score [0.0, 2.0]."
)
```

**`cache_init_params.py`** — new fields:
```python
@dataclass
class CacheInitParams:
    eviction_policy: str = "lru"       # "lru" | "marconi"
    eff_weight: float = 0.5            # α in Marconi paper
    model_config: Optional[Any] = None # Qwen3NextConfig (or compatible)
    cache_params: Optional[Any] = None # Mamba2CacheParams
```

**`mamba_radix_cache.py`** — wire into `__init__`:
```python
class MambaRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.eviction_policy = params.eviction_policy
        self.eff_weight = params.eff_weight
        self.model_config = params.model_config
        self.cache_params = params.cache_params
```

### 6.2 Extend TreeNode

Add fields to `TreeNode` for Marconi scoring:

```python
class TreeNode:
    # sequence length at this node for computing FLOPs saved relative to parent
    num_cached_tokens: int = 0

    # cached FLOP efficiency score (FLOPs_saved / bytes_used), invalidated on access
    _flop_efficiency: Optional[float] = None
```

### 6.3 Add FLOP Utility Functions

New file `python/sglang/srt/mem_cache/marconi_utils.py` with FLOP/memory calculations for Qwen3-Next:

```python
def get_full_attn_flops(seqlen, hidden_size, num_kv_heads, head_dim) -> float:
    """FLOPs for one full-attention layer (GQA) on seqlen tokens."""
    proj_flops = 8 * seqlen * hidden_size ** 2
    attn_flops = 4 * seqlen ** 2 * hidden_size
    return proj_flops + attn_flops

def get_linear_attn_flops(seqlen, num_heads, head_dim, state_size) -> float:
    """FLOPs for one Mamba2 / linear-attention layer."""
    ssm_scan_flops = 2 * seqlen * num_heads * head_dim * state_size
    proj_flops = 12 * seqlen * (num_heads * head_dim) ** 2
    return ssm_scan_flops + proj_flops

def get_moe_flops(seqlen, hidden_size, num_experts_per_tok, expert_intermediate_size) -> float:
    """FLOPs for one MoE FFN layer on seqlen tokens.
    MoE FLOPs are included in the efficiency formula: they are genuinely saved by
    prefix caching and constitute the dominant term for large MoE models like Qwen3-Next.
    """
    return 4 * seqlen * hidden_size * expert_intermediate_size * num_experts_per_tok

def compute_flops_saved(prefix_len, total_len, config) -> float:
    """Total FLOPs saved by reusing a cached prefix of prefix_len tokens,
    for a request of estimated total_len tokens (= 2 * prefix_len by default).
    Marginal attention savings are computed as attn_flops(total_len) - attn_flops(total_len - prefix_len)
    to account for the quadratic benefit of skipping tokens in a longer context.
    Includes full-attention, Mamba2/linear-attention, and MoE FFN layers.
    """
    # ... sums across all layer types using config fields ...

def compute_memory_bytes(prefix_len, cache_params, config, model_config, tp_world_size) -> float:
    """Total per-GPU memory for one cached node: SSM state (fixed) + KV cache (proportional to prefix_len).
    Both terms are TP-sharded (per-GPU bytes):
    - mamba_cache_per_req is already sharded via Mamba2StateShape.create(tp_world_size=...)
    - get_num_kv_heads(tp_world_size) uses max(1, total // tp) to handle head replication
    """
    ssm_bytes = cache_params.mamba_cache_per_req
    num_kv_heads_per_gpu = model_config.get_num_kv_heads(tp_world_size)
    kv_bytes = prefix_len * num_kv_heads_per_gpu * head_dim * 2 * kv_dtype_bytes * num_attn_layers
    return ssm_bytes + kv_bytes

def compute_flop_efficiency(prefix_len, total_len, cache_params, config, model_config, tp_world_size) -> float:
    """Marconi efficiency score: FLOPs_saved / memory_bytes."""
    return compute_flops_saved(prefix_len, total_len, config) / (
        compute_memory_bytes(prefix_len, cache_params, config, model_config, tp_world_size) + 1e-8
    )
```

### 6.4 Implement Marconi FLOP-Aware Cache Eviction Policy

Both `evict_mamba()` (SSM pressure) and `evict()` (KV pressure) are updated. When KV is evicted, SSM is freed simultaneously per the invariant in §2.4.

```python
def evict_mamba(self, mamba_num: int) -> int:
    if self.eviction_policy == "marconi" and self.model_config is not None:
        return self._evict_mamba_marconi(mamba_num)
    return self._evict_mamba_lru(mamba_num)   # existing behavior renamed
```

Add `_evict_mamba_marconi()` — FLOP-aware eviction:
1. Collect all unlocked candidates from `mamba_lru_list` — **leaf and internal nodes with any number of children**
2. For each candidate, compute `efficiency_score` via `compute_flop_efficiency(prefix_len=node.num_cached_tokens, total_len=2*node.num_cached_tokens, ...)`
3. Compute `recency_score = 1 / (current_ts - node.last_access_time)`
4. Normalize both scores to [0, 1] with min-max normalization (degenerate case: all equal → uniform scores, falls back to recency-only)
5. Compute `utility = eff_weight * normalized_efficiency + normalized_recency`
6. Evict the candidate with the lowest utility:
   - `len(node.children) == 0` (leaf): `_evict_leaf_node()` — frees both SSM and KV per §2.4
   - `len(node.children) > 0` (internal): `_tombstone_internal_node()` — frees SSM only, KV kept

### 6.5 Timestamp Update Optimization

In `_match_prefix_helper`, only update `last_access_time` on the terminal node instead of every ancestor:

```python
if self.eviction_policy == "marconi":
    last_matched_node.last_access_time = get_last_access_time()
    last_matched_node._flop_efficiency = None  # Invalidate cached score
else:
    # Existing behavior: update all ancestors
    node = last_matched_node
    while node != self.root_node:
        node.last_access_time = get_last_access_time()
        node = node.parent
```

## 7. Testing Plan

### Unit Tests

1. **FLOP utility tests** (`test_marconi_utils.py`):
   - Validate `get_full_attn_flops`, `get_linear_attn_flops`, `get_moe_flops` against paper formulas
   - Verify `compute_memory_bytes` matches `mamba_cache_per_req` for standard configs
   - Verify `compute_memory_bytes` with `tp_world_size > 1` correctly shards KV head count
   - Test edge cases: `prefix_len = 0`, `prefix_len = total_len`

2. **Eviction policy tests** (`test_mamba_radix_cache_marconi.py`):
   - Insert entries with known efficiency profiles; verify higher-efficiency entries survive eviction
   - Verify LRU behavior is unchanged when `eviction_policy="lru"`
   - Test intermediate node eviction (tombstoning) with Marconi scoring
   - Test score normalization edge cases (all equal efficiencies → recency-only)
   - Verify SSM state is freed whenever KV cache is evicted (no orphaned SSM state)

3. **Integration tests**:
   - Full `cache_finished_req` / `cache_unfinished_req` / `match_prefix` cycle with `eviction_policy="marconi"`
   - Verify `mamba_cache_per_req` bytes accounting remains consistent

### Benchmarks

Metrics to track against baseline LRU:

**SSM state hit rate**: ≥ 2× improvement (paper shows 4.5–34.4× on LMSys)
**TTFT for cacheable requests**: ≥ 10% reduction
**TTFT regression on non-cacheable**: < 1% (should be zero)
**Eviction overhead (latency)**: < 1ms per eviction call

---
