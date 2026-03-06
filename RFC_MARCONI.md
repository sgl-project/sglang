# RFC: Marconi Prefix Caching for Hybrid LLMs in SGLang

Related Paper: [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/pdf/2411.19379)

Related Code: https://github.com/ruipeterpan/marconi/tree/main 

---

## 1. Summary

This RFC proposes integrating Marconi's prefix caching into SGLang. We extend the SGLang's existing support for Hybrid Architecture with Marconi's cache admission and eviction policy.

Target Model: `Qwen/Qwen3-Next-80B-A3B-Instruct`

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

`MambaRadixCache` holds a radix tree of `TreeNode`s, two independent LRU lists, and references to the index allocators:

```python
class MambaRadixCache(BasePrefixCache):
    root_node: TreeNode                                    # root

    # Free list tracker to store which slots are in use vs available
    req_to_token_pool: HybridReqToTokenPool                # free pages management for SSM slots
    token_to_kv_pool_allocator: TokenToKVPoolAllocator     # free_pages management for KV slots

    # Two LRU lists for eviction ordering
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

## 3. Scope

We will be adding a scoring layer into the eviction decision of `MambaRadixCache`.

### In Scope

- **Cache admission policy**: 
  - **Input-only**: Track incoming request prefixes to identify hot common prefixes; only cache SSM states at nodes with high reuse likelihood
  - **Input-and-output**: Only cache SM states that represent the last decoded token between conversation rounds
- **SSM state eviction**: Replace pure LRU with FLOP-efficiency-weighted scoring in `evict_mamba()`
- **FLOP utility functions**: New `marconi_utils.py` with FLOP/memory calculations for Qwen3-Next
- **Timestamp update**: Update only the terminal node on prefix match (not all ancestors)

## 4. API Summary
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --eviction-policy marconi \
    --marconi-eff-weight 0.7 \
    --marconi-min-reuse-count 2
```
### 5. Backward Compatibility

- Default `eviction_policy="lru"` 
- No changes to `MambaRadixCache`'s public API (`match_prefix`, `insert`, `evict`, `cache_finished_req`, `cache_unfinished_req`)
- All existing hybrid models (Falcon-H1, Nemotron-H, LFM2, etc.) continue to use LRU

## 6 Changes Required

### 6.1 Add Server Args and CacheInitParams

**`server_args.py`** — new CLI arguments:
```python
# existing
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
parser.add_argument(
    "--marconi-min-reuse-count",
    type=int, default=2,
    help="Minimum prefix reuse count before caching SSM state."
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

    # cached FLOP efficiency score (FLOPs_saved / bytes_used)
    _flop_efficiency: Optional[float] = None

    # track how often each prefix is seen
    prefix_frequency: dict[tuple, int]
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
    """FLOPs for one MoE FFN layer on seqlen tokens."""
    return 4 * seqlen * hidden_size * expert_intermediate_size * num_experts_per_tok

def compute_flops_saved(prefix_len, total_len, config) -> float:
    """Total FLOPs saved by reusing a cached prefix."""
    # ... sums across all layer types using config fields ...

def compute_memory_bytes(prefix_len, cache_params, config) -> float:
    """Total memory: SSM state (fixed) + KV cache (proportional to prefix_len)."""
    ssm_bytes = cache_params.mamba_cache_per_req  # from BaseLinearStateParams
    kv_bytes = prefix_len * num_kv_heads * head_dim * 2 * kv_dtype_bytes * num_attn_layers
    return ssm_bytes + kv_bytes

def compute_flop_efficiency(prefix_len, total_len, cache_params, config) -> float:
    """Marconi efficiency score: FLOPs_saved / memory_bytes."""
    return compute_flops_saved(...) / (compute_memory_bytes(...) + 1e-8)
```

### 6.4 Implement Marconi FLOP-Aware Cache Eviction Policy

```python
def evict_mamba(self, mamba_num: int) -> int:
    if self.eviction_policy == "marconi" and self.model_config is not None:
        return self._evict_mamba_marconi(mamba_num)
    return self._evict_mamba_lru(mamba_num)   # existing behavior renamed
```

Add `_evict_mamba_marconi()` — FLOP-aware eviction:
1. Collect all unlocked candidates from `mamba_lru_list`
2. Compute `efficiency_score` (via `compute_flop_efficiency`) and `recency_score`
4. Evict the candidate with the lowest `eff_weight * efficiency + recency`
5. Evict via existing `_evict_leaf_node()` or tombstone internal nodes

### 6.5 Implement Cache Admission Policy

**Input-only admission** — Only cache SSM states for prefixes likely to be reused:
- In `cache_unfinished_req()`, check frequency before calling `mamba_pool.fork_from()` 

**Input-and-output admission** — Only cache SSM states that capture complete conversation turns:
- In `cache_finished_req()`: current behaviors cache SSM state at the leaf node, no change needed
- The `_split_node()` path (branchoff): Only cache SSM state this if the branchoff node's prefix has high reuse frequency

### 6.6 Timestamp Update Optimization

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

## 6. Testing Plan

### Unit Tests

1. **FLOP utility tests** (`test_marconi_utils.py`):
   - Validate `get_full_attn_flops`, `get_linear_attn_flops`, `get_moe_flops` against paper formulas
   - Verify `compute_memory_bytes` matches `mamba_cache_per_req` for standard configs
   - Test edge cases: `prefix_len = 0`, `prefix_len = total_len`

2. **Eviction policy tests** (`test_mamba_radix_cache_marconi.py`):
   - Insert entries with known efficiency profiles; verify higher-efficiency entries survive eviction
   - Verify LRU behavior is unchanged when `eviction_policy="lru"`
   - Test intermediate node eviction (tombstoning) with Marconi scoring
   - Test score normalization edge cases (all equal efficiencies → recency-only)

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

## 7. Open Questions

1. **KV cache eviction**: Should we also apply Marconi-style scoring to KV cache? 

2. **eff-weight**: Regarding adaptive config tunning mentioned in the paper, this code seems to be out of scope for sglang? We're adding a cli arg for eff-weight and default it as 0.5, and user can tune this parameter and set it accordingly

3. **`total_len` estimation**: To compute Marconi's flop efficiency score, we need an estimate of future request length. Currently I'm using a 2*cached prefix length as a default, should we consider tracking a rolling average of the actual request lengths? and should this tracked globally across all requests, or per radix tree node

4. **MoE FLOP treatment**: In the efficiency formula, MoE FFN FLOPs are included. Since MoE FLOPs dominate for large models, should MoE FLOPs be excluded from the efficiency formula?

---