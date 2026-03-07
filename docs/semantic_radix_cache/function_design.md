# Semantic-Aware KV Cache Pruning for SGLang Radix Cache

## Overall Goal

Enhance SGLang's radix cache implementation to add **semantic awareness** for handling **non-linear context patterns** in agent workflows. Specifically, when a summary request is made, the KV cache entries for previous conversation turns should be efficiently pruned, as they become obsolete after summarization.

## Architecture Overview

### Original Flow
```
[System][Q1][A1][Q2][A2][...][Summary Q][Summary A]
                     ↓
              Radix Cache
                     ↓
After summary: Only [System] remains
```

### Target Flow
1. Client sends request with `semantic_event="start"` → Auto-creates session
2. Multiple conversation rounds accumulate in KV cache
3. Client sends request with `semantic_event="reset"` → Prunes old history
4. New conversation starts fresh

---

## Files Modified

### 1. Core Data Structures

| File | Changes |
|------|---------|
| `sglang/srt/managers/io_struct.py` | Added `semantic_event` field to `GenerateReqInput`, `TokenizedGenerateReqInput`; Removed `semantic_event` from `SessionParams` |
| `sglang/srt/entrypoints/openai/protocol.py` | Added `semantic_event` field to request models |

### 2. OpenAI API Endpoints

| File | Changes |
|------|---------|
| `sglang/srt/entrypoints/openai/serving_completions.py` | Pass `semantic_event` to internal request |
| `sglang/srt/entrypoints/openai/serving_chat.py` | Pass `semantic_event` to internal request |

### 3. Tokenizer & Request Processing

| File | Changes |
|------|---------|
| `sglang/srt/managers/tokenizer_manager.py` | Fixed `session_params` handling; Empty dict `{}` now creates `SessionParams()` instead of `None` |

### 4. Scheduler & Session Management

| File | Changes |
|------|---------|
| `sglang/srt/managers/scheduler.py` | Auto-create session when `semantic_event` provided; Pass `semantic_event` to `Req` object; Return `session_id` in response metadata |

### 5. Output Processing (Critical)

| File | Changes |
|------|---------|
| `sglang/srt/managers/scheduler_output_processor_mixin.py` | Added semantic event handling in prefill & decode batch processing; Added pruning logic for already-finished requests |

### 6. Radix Cache

| File | Changes |
|------|---------|
| `sglang/srt/mem_cache/radix_cache.py` | Added `prune_from_node()` method; Added `_node_exists()` method for safety checks; Added double-free protection, root protection, and child constraint checks |

### 7. Scheduling Policy

| File | Changes |
|------|---------|
| `sglang/srt/managers/schedule_policy.py` | Added `_prioritize_reset_requests()` method to prioritize reset requests in waiting queue; Modified `calc_priority()` to call prioritization at the beginning |

### 8. Test Scripts

| File | Changes |
|------|---------|
| `sglang/test/test_kv_cache_session.py` | Comprehensive test with KV cache monitoring |
| `sglang/test/test_session_params.py` | Session params propagation test |
| `sglang/test/test_kv_cache_session_openai_client.py` | OpenAI client test for session management |
| `sglang/test/test_kv_cache_session_concurrent.py` | Concurrent request handling test |
| `sglang/test/test_reset_priority_scheduling.py` | Reset request prioritization test |

---

## Problems Encountered & Solutions

### Problem 1: Session ID Returned as List Instead of String

**Symptom**: 
```
Session ID: ['eb8e56cbe7a24df2a79c445e4da88b57']
```

**Root Cause**: In `scheduler_output_processor_mixin.py`, the code does:
```python
# Line 1078
customized_info[k].append(v)  # v is already [session_id], so it becomes [[session_id]]
```
Then in `tokenizer_manager.py`:
```python
meta_info[k] = v[i]  # v[i] = [[session_id]][0] = [session_id]
```

**Solution**: Changed scheduler.py to pass string directly:
```python
req.customized_info = {"session_id": auto_created_session_id}  # Not in a list
```

---

### Problem 2: Empty `session_params: {}` Treated as None

**Symptom**: Auto-creation not triggered when `session_params: {}` sent

**Root Cause**: In `tokenizer_manager.py`:
```python
session_params = SessionParams(**session_params_data) if session_params_data else None
# Empty dict {} is falsy in Python, so this returns None
```

**Solution**: Changed condition to check for `None` explicitly:
```python
session_params = SessionParams(**session_params_data) if session_params_data is not None else None
```

---

### Problem 3: Auto-Creation Not Triggered with Just `semantic_event`

**Symptom**: Requests with `semantic_event="start"` but no `session_params` didn't auto-create

**Root Cause**: The auto-creation logic only checked:
```python
if recv_req.session_params is not None and recv_req.session_params.id is None:
```

**Solution**: Added condition to auto-create when `semantic_event` is provided:
```python
should_auto_create = (
    (recv_req.session_params is not None and recv_req.session_params.id is None)
    or (semantic_event_debug is not None and recv_req.session_params is None)
)
```

---

### Problem 4: Pruning Logic Only in Prefill, Not Decode

**Symptom**: Tree kept growing after summary; no pruning logs

**Root Cause**: Semantic event handling was only in `process_batch_result_prefill()`, but requests can finish during decode phase

**Solution**: Added semantic event handling to `process_batch_result_decode()`:
```python
if req.finished():
    semantic_event = getattr(req, 'semantic_event', None)
    is_insert = not (semantic_event == 'reset')
    release_kv_cache(req, self.tree_cache, is_insert=is_insert)
    
    if semantic_event == 'reset':
        self.tree_cache.prune_from_node(req.last_node)
```

---

### Problem 5: Already-Finished Requests Skipped Pruning

**Symptom**: Short requests (like summary with few tokens) skip pruning

**Root Cause**: In prefill processing:
```python
if req.finished() or req.is_retracted:
    continue  # Skips all pruning logic!
```

**Solution**: Added pruning before the `continue`:
```python
if req.finished() or req.is_retracted:
    if req.finished():
        semantic_event = getattr(req, 'semantic_event', None)
        if semantic_event == 'reset':
            if hasattr(req, 'last_node') and req.last_node is not None:
                self.tree_cache.prune_from_node(req.last_node)
    continue
```

---

### Problem 6: IndexError in Tokenizer Manager (Concurrent Requests)

**Symptom**: 
```
IndexError: list index out of range at tokenizer_manager.py:1555
```

**Root Cause**: Metadata list length mismatch due to list wrapping. When `customized_info[k].append(v)` is called where `v` is a list, it creates nested lists. Then accessing `v[i]` fails when the list structure is incorrect.

**Solution**: Ensure all requests have entries in `customized_info` even if `None`. Fixed by:
1. Properly initializing `customized_info` for all requests
2. Ensuring consistent data types (strings, not lists, for simple values like `session_id`)

---

### Problem 7: Race Condition with Concurrent Reset Requests

**Symptom**: Limited pruning when multiple reset requests finish in the same batch. Tree size doesn't decrease as expected.

**Root Cause**: When multiple reset requests share the same parent node, each request sees a stale `lock_ref` value:
```
Batch with 3 reset requests: A, B, C
All share parent node P (lock_ref=3)

Request A finishes:
- release_kv_cache(A) → lock_ref(P) = 2
- prune_from_node(A) → sees lock_ref(P)=2, stops!

Request B finishes:
- release_kv_cache(B) → lock_ref(P) = 1
- prune_from_node(B) → sees lock_ref(P)=1, stops!

Request C finishes:
- release_kv_cache(C) → lock_ref(P) = 0
- prune_from_node(C) → prunes successfully
```

**Solution**: Implemented deferred batch pruning with three phases:
1. **Collect**: Gather all reset requests that finished in the batch
2. **Unlock**: Release locks for all collected requests (decrement `lock_ref`)
3. **Prune**: Perform pruning for all collected requests

Implementation in `scheduler_output_processor_mixin.py`:
```python
# Phase 1: Collect reset requests
reset_requests_to_prune = []
for req in batch.reqs:
    if req.finished():
        semantic_event = getattr(req, 'semantic_event', None)
        if semantic_event == 'reset' and hasattr(req, 'last_node'):
            reset_requests_to_prune.append((req, req.last_node))

# Phase 2: Release locks
for req, _ in reset_requests_to_prune:
    release_kv_cache(req, self.tree_cache, is_insert=False)

# Phase 3: Prune
for req, node in reset_requests_to_prune:
    self.tree_cache.prune_from_node(node)
```

---

### Problem 8: Double-Free and Root Protection in Pruning

**Symptom**: Potential crashes when pruning already-deleted nodes or attempting to prune root nodes

**Root Cause**: No safety checks in `prune_from_node()` method

**Solution**: Added comprehensive safety checks:
```python
def prune_from_node(self, start_node: TreeNode):
    # Check if node still exists
    if not self._node_exists(start_node):
        logger.debug(f"Node {start_node} already deleted, skipping")
        return
    
    # Protect root and its immediate children
    if start_node == self.root_node or start_node.parent == self.root_node:
        logger.warning("Protecting root or its immediate child")
        return
    
    # Check child constraints before pruning
    if start_node.children:
        logger.warning(f"Node {start_node} has children, cannot prune")
        return
    
    # ... proceed with pruning ...
```

---

## Final Results

### Before Fixes
- Session ID: `['eb8e56cbe7a24df2a79c445e4da88b57']` (list)
- Tree size after summary: 130 nodes (growing!)
- No pruning logs visible
- IndexError in concurrent requests
- Race conditions with multiple reset requests

### After Fixes
- Session ID: `eb8e56cbe7a24df2a79c445e4da88b57` (string) ✓
- Tree size before summary: 104 nodes
- Tree size after summary: 6 nodes ✓
- **Pruned 98 nodes!** ✓
- Logs show: `is_insert=False`, `Pruning complete. Pruned 5 nodes.` ✓
- **No IndexError in concurrent requests** ✓
- **Deferred batch pruning eliminates race conditions** ✓
- **Safety checks prevent double-free and root deletion** ✓
- **Reset requests prioritized in scheduling** ✓

---

## Performance Impact

### Reset Request Prioritization
- **Eviction rate**: Lower (more efficient cache management)
- **Average TTFT**: Decreased (faster time to first token)
- **Available slots**: 4% → 3% (time-averaged, tighter utilization)
- **Evictable ratio**: 8.99% → 8.76% (more aggressive pruning)
- **New prefill tokens**: 16.9% decrease (more cache hits)

### Key Insights
1. **Reset prioritization** improves memory utilization by cleaning cache faster
2. **Deferred batch pruning** ensures all reset requests in a batch are processed correctly
3. **Safety checks** prevent crashes while maintaining tree integrity
4. **Memory-aware scheduling** opportunities exist to further optimize throughput

---

## Key Learnings

1. **Request finish points vary**: Short requests can finish at different pipeline stages (prefill, decode, or before output processing)

2. **Empty collections are falsy**: Empty dict `{}` and empty list `[]` are falsy in Python, causing subtle bugs

3. **List wrapping issue**: When collecting data across requests, values get wrapped in extra layers of lists

4. **Semantic events need handling at multiple points**: Both prefill and decode processing need semantic event logic

5. **Race conditions in concurrent pruning**: Multiple reset requests sharing parent nodes can see stale `lock_ref` values, leading to limited pruning. Solution: deferred batch pruning (collect, unlock, prune)

6. **Safety checks are critical**: Pruning operations need comprehensive checks for node existence, root protection, and child constraints to prevent crashes and data corruption

7. **Prioritization improves throughput**: Reset requests should be prioritized to free memory faster, enabling better utilization for subsequent requests

8. **Memory-aware scheduling opportunities**: Traditional FCFS can be enhanced with memory awareness to better utilize freed space from pruning operations

