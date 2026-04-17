# KV Connector Guide

Implement `BaseKVConnector` (`sglang/srt/mem_cache/kv_connector.py`) to integrate an external KV cache offloading framework into SGLang.

## Architecture

```
Scheduler
    │
    v
ExtendedRadixCache  ───>  RadixCache (manage GPU KV pool)
    │
    v
BaseKVConnector (your impl)  ───>  External Storage (CPU/SSD/remote)
```

`ExtendedRadixCache` wraps `RadixCache` via composition. It intercepts
`match_prefix`, `cache_finished_req` etc. to coordinate with the connector.
GPU memory allocation and radix tree operations are handled by the inner
`RadixCache`; the connector only manages external storage and async transfers.

## How It Works

Each scheduler iteration:

1. **Poll completions** — `check_completed_store_tasks()` / `check_completed_load_tasks()`
   release GPU radix tree node locks so GPU nodes become evictable, otherwise tree nodes stay locked and GPU memory leaks.
2. **Match prefix** — `get_new_hit_length()`: query external storage for tokens
   beyond the GPU radix cache hit.
3. **Prepare load** — `init_load_back`: pre-allocate GPU slots, create locked
   TreeNode, enqueue `LoadOperation`.
4. **Dispatch load** — `start_load_kv()`: async transfer from external storage into pre-allocated GPU slots.
5. **Request completes** — `start_store_kv()`: async transfer of new KV data to external storage; node locked until done.

## Constructor

```python
class BaseKVConnector(ABC):
    def __init__(self, params, server_args, tp_group, tp_rank, kvcache):
```

| Parameter     | Key Contents                                                       |
|---------------|--------------------------------------------------------------------|
| `params`      | `page_size`, `token_to_kv_pool_allocator`                          |
| `server_args` | Model path, `tp_size`, dtype, `kv_connector_extra_config`          |
| `tp_group`    | `torch.distributed` process group (`None` if tp=1)                 |
| `tp_rank`     | 0-based TP rank                                                    |
| `kvcache`     | `k_buffer` / `v_buffer`: list of per-layer GPU tensors `[num_slots, num_kv_heads, head_dim]` |

## Methods to Implement

### Abstract (required)

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_new_hit_length` | `(token_ids, token_mask, update_state_for_load, rid) -> int` | Query external storage for cached tokens. `token_mask[i]=True` for positions not on GPU. When `update_state_for_load=True`, reserve state keyed by `rid` for the upcoming load. |
| `release_load_state` | `(rid) -> None` | Release state reserved by `get_new_hit_length` (called when GPU alloc fails). |
| `start_load_kv` | `(task_id, load_ops: List[LoadOperation]) -> None` | Begin async KV cache transfer from storage to GPU. Each `LoadOperation` has `rid`, `device_indices` (pre-allocated GPU slots), and `node` (opaque). |
| `check_completed_load_tasks` | `() -> List[int]` | Non-blocking poll. Return completed load `task_id`s. |
| `start_store_kv` | `(task_id, token_ids, kv_indices: Tensor) -> None` | Begin async KV cache transfer from GPU to storage. `kv_indices[i]` is the GPU pool slot for `token_ids[i]`. |
| `check_completed_store_tasks` | `() -> List[int]` | Non-blocking poll. Return completed store `task_id`s. |

### Optional (default no-ops)

| Method | Purpose |
|--------|---------|
| `prefetch(rid, token_ids)` | Pre-stage data from slow to fast tier before scheduling. |
| `check_prefetch_progress(rid) -> bool` | Gate scheduling (default `True`). |
| `pop_prefetch_loaded_tokens(rid) -> int` | Pop and return the number of tokens loaded from storage for a request. Returns 0 if no prefetch was done or was revoked. This should be called after check_prefetch_progress() returns True. |
| `cancel_prefetch(rid)` | Cancel prefetch on request abort. |
| `layer_done_counter` (property) | Layer-wise sync counter for overlapped load+compute. |
| `reset()` | Clear all state on cache flush. |
| `shutdown()` | Cleanup resources. |

## Example

Reference implementation: `FlexKVConnector` (`sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py`).

## Launch

```bash
# Full module path
python -m sglang.launch_server --model ... \
    --kv-connector-cls my_package.my_module.MyConnector

# Built-in alias
python -m sglang.launch_server --model ... \
    --kv-connector-cls flexkv
```
