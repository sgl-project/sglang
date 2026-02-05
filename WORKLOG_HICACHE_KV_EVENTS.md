# HiCache + Dynamo Router Integration Worklog

**Project:** [SGL Hi-Cache + Dynamo Router integration](https://linear.app/nvidia/project/sgl-hi-cache-dynamo-router-integration-38651e3b3f9c)

**Date:** 2026-02-04

---

## Completed Tickets

### LLM-81: Add `medium` field to KV event types

**Files Modified:**
- `python/sglang/srt/disaggregation/kv_events.py`

**Changes:**
- Added vLLM-compatible storage tier constants:
  ```python
  MEDIUM_GPU = "GPU"
  MEDIUM_CPU_TIER1 = "CPU_TIER1"
  MEDIUM_CPU_TIER2 = "CPU_TIER2"
  ```
- Added `medium: Optional[str] = None` field to `BlockStored` class
- Added `medium: Optional[str] = None` field to `BlockRemoved` class

---

### LLM-82: Add `medium="GPU"` to existing L1 cache events

**Files Modified:**
- `python/sglang/srt/mem_cache/radix_cache.py`

**Changes:**
- Updated `_record_store_event()` to emit `BlockStored(medium=MEDIUM_GPU)`
- Updated `_record_remove_event()` to emit `BlockRemoved(medium=MEDIUM_GPU)`

---

### LLM-83: Emit KV events for L1<->L2 tier transitions

**Files Modified:**
- `python/sglang/srt/mem_cache/hiradix_cache.py`

**Changes:**
- Added imports for `MEDIUM_GPU`, `MEDIUM_CPU_TIER1`, `BlockStored`, `BlockRemoved`, and `hash_str_to_int64`
- Added `_record_tier_store_event(node, medium)` helper method for BlockStored events
- Added `_record_tier_remove_event(node, medium)` helper method for BlockRemoved events
- Updated `write_backup()` to emit:
  - `BlockStored(medium="CPU_TIER1")` - data stored in host memory (block still in GPU too)
- Updated `_evict_backuped()` to emit:
  - `BlockRemoved(medium="GPU")` - GPU memory freed for backed-up node
- Updated `_evict_regular()` to emit:
  - `BlockRemoved(medium="GPU")` - GPU memory freed for non-backed-up node
- Updated `load_back()` to emit:
  - `BlockStored(medium="GPU")` - data loaded back to GPU
- Updated `evict_host()` to emit:
  - `BlockRemoved(medium="CPU_TIER1")` - data removed from host memory

**Key Design Decision:**
- Blocks can exist in multiple tiers simultaneously (e.g., GPU + Host after write_backup)
- `BlockRemoved(GPU)` is emitted when GPU memory is actually freed (in evict functions)
- NOT when data is copied to host (write_backup just adds another tier)

**Implementation Pattern:**
- Follows the same pattern as `radix_cache.py` for event recording
- Uses `enable_kv_cache_events` flag to guard event emission
- Computes hash values lazily if not already set
- Events include correct block hashes and token IDs

---

### LLM-84: Emit KV events for L2<->L3 tier transitions

**Files Modified:**
- `python/sglang/srt/mem_cache/hiradix_cache.py`

**Changes:**
- Added import for `MEDIUM_CPU_TIER2`
- Added `_record_prefetch_store_events()` helper method for prefetch data events
- Updated `write_backup_storage()` to emit:
  - `BlockStored(medium="CPU_TIER2")` - data stored in remote storage (L3)
- Updated `check_prefetch_progress()` to emit:
  - `BlockStored(medium="CPU_TIER1")` - data loaded from storage to host memory

**Notes on L3 Eviction:**
- L3 eviction (`BlockRemoved(medium="CPU_TIER2")`) is NOT implemented yet
- Current storage backends (Mooncake, 3FS, NIXL, AIBrix) handle their own eviction
- No explicit L3 eviction callback exists in `hiradix_cache.py`
- This may require coordination with storage backend implementations

---

## Pull Request

**PR:** https://github.com/sgl-project/sglang/pull/18205

**Branch:** `idhanani/llm-81-sglang-add-medium-field-to-kv-event-types`

**Status:** Open, awaiting review

**Notes:**
- Gemini Code Assist suggested using Enum instead of constants
- Declined: vLLM uses simple string constants, not Enums, and we need API compatibility for Dynamo's parser

---

### Bugfix: Missing BlockStored(GPU) events in HiRadixCache.insert()

**Date:** 2026-02-04

**Files Modified:**
- `python/sglang/srt/mem_cache/hiradix_cache.py`

**Problem:**
KV event correctness validation was failing with 157 "orphan remove" errors. `BlockRemoved(GPU)` events were being emitted for blocks that never had a corresponding `BlockStored(GPU)` event.

**Root Cause:**
`HiRadixCache.insert()` completely overrides `RadixCache.insert()` but did NOT emit `BlockStored(GPU)` events when inserting new blocks. The parent class calls `_record_store_event()` in `_insert_helper()`, but `HiRadixCache` reimplements the insert logic without this call.

**Fix:**
Added `self._record_tier_store_event(node, MEDIUM_GPU)` calls in three locations within `HiRadixCache.insert()`:

1. **Line ~1505**: When a previously evicted node gets its value restored
2. **Line ~1519**: When splitting a node and the new split node was evicted
3. **Line ~1543**: When inserting a completely new node

**Validation:**
- Before fix: 157 errors ("orphan remove: block not in GPU tier")
- After fix: 0 errors, VALIDATION PASSED

**Test Results:**
```
Event Statistics:
   Total batches:    13
   Total events:     605
   BlockStored:      456
   BlockRemoved:     149
   AllBlocksCleared: 0
   Unique blocks:    149

Events by Medium:
   remove_GPU: 149
   store_CPU_TIER1: 149
   store_CPU_TIER2: 149
   store_GPU: 158
```

---

## Testing Performed

1. **Unit tests:** 35 tests passed in `test_kv_events.py`
2. **Msgpack serialization:** Verified events serialize/deserialize correctly
3. **Live server test:** Confirmed 2062 BlockStored events emitted with `medium='GPU'`
4. **KV Event Correctness:** `test_kv_events_correctness.py` passes with 0 errors

---

## Key Learnings

- Current implementation only emits events from `radix_cache.py` (L1/GPU tier)
- HiCache tier transitions (L1<->L2, L2<->L3) are handled in `hiradix_cache.py`
- `hiradix_cache.py` does NOT currently emit KV events for tier transitions
- This is expected per project scoping - L2/L3 events are separate tickets

---

## Remaining Tickets

| Ticket | Description | File |
|--------|-------------|------|
| LLM-85 | Verify Dynamo parsing | Dynamo repo |
| LLM-86 | E2E integration test | Both repos |

**Note:** LLM-84 L3 eviction events are partially complete - L3 eviction callback needs to be added to storage backends.

---

## Reference Files

- SGLang KV events: `python/sglang/srt/disaggregation/kv_events.py`
- SGLang radix cache: `python/sglang/srt/mem_cache/radix_cache.py`
- SGLang hiradix cache: `python/sglang/srt/mem_cache/hiradix_cache.py`
- vLLM reference: `/home/ubuntu/vllm/vllm/distributed/kv_events.py`
- Dynamo parser: `lib/llm/src/block_manager/kv_consolidator/tracker.rs`
