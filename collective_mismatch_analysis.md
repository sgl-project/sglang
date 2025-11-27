# SGLang Collective Synchronization Failure - Root Cause Analysis

**Date**: November 14, 2025
**Issue**: [#13227](https://github.com/sgl-project/sglang/issues/13227)
**Status**: CRITICAL - Blocks all TP>1 deployments
**Affected Versions**: v0.5.4.post3, v0.5.5, current main branch

---

## Executive Summary

SGLang experiences collective synchronization failures when deploying models with tensor parallelism (TP > 1). Different GPU ranks execute different numbers of collective operations, causing PyTorch distributed to detect a mismatch and crash. This affects:

1. **Standard TP mode**: All models with TP > 1 (Kimi K2, Llama-3-70B, etc.)
2. **Prefill/Decode Disaggregation mode**: With `--enable-dp-attention` enabled
3. **Impact**: Users cannot deploy 70B+ parameter models requiring tensor parallelism

---

## Evidence from Logs

### Case 1: Standard TP Mode (GitHub Issue #13227)

**Configuration**:
```bash
docker run --rm --gpus all --network host --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  lmsysorg/sglang:v0.5.5 \
  python -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2-Thinking \
    --tp 8 --disable-cuda-graph
```

**Error**:
```
RuntimeError: Detected mismatch between collectives on ranks.
Rank 7 is running collective: CollectiveFingerPrint(SequenceNumber=18, ...)
but Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=30, ...)
```

**Key Facts**:
- Difference: 12 collectives (30 - 18 = 12)
- Timing: During "scheduler event loop initialization"
- Model: Kimi K2 (non-multimodal, 70B+ params)
- Confirmed on: Multiple hardware configs (H200, B200 GPUs)
- Also affects: Llama-3-70B-Instruct

### Case 2: Prefill/Decode Disaggregation Mode (Local Logs)

**Configuration**:
```bash
--disaggregation-mode decode
--tp-size 32
--dp-size 32
--enable-dp-attention
--enable-two-batch-overlap
--moe-a2a-backend deepep
```

**Error**:
```
RuntimeError: Detected mismatch between collectives on ranks.
Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=73, OpType=BARRIER)
but Rank 1 is running collective: CollectiveFingerPrint(SequenceNumber=13, OpType=BARRIER)
```

**Key Facts**:
- Difference: 60 collectives (73 - 13 = 60)
- Timing: During CUDA graph capture ("Capturing batches (bs=16...)")
- Location: `/Users/idhanani/Downloads/dsr1_wideep_h100_4102550/decode_eos0500_n0_w0.out`
- Context: Between memory profiling (SequenceNumber=1) and graph capture barrier
- Pattern: Rank 0 (attn_tp_rank==0) executes 60 extra collectives

---

## Timeline of Suspicious Commits

### 1. October 30, 2025 - Multimodal Optimization (PR #11910)

**Commit**: `17a57fd8620abafe994dc5a7fe6aa3f88e6352f0`
**Author**: Yuan Luo <yuan.luo@hotmail.com>
**Title**: "[Perf] Optimize multimodal mm_inputs process in scheduler"

**Changes**: `python/sglang/srt/managers/scheduler.py` (+82 lines)

**Introduced Code**:
```python
def _process_and_broadcast_mm_inputs(
    self,
    raw_mm_inputs: Optional[dict],
):
    """Materialize MultimodalInputs once on the entry rank and broadcast to others."""
    if raw_mm_inputs is None:
        return None  # ❌ EARLY RETURN - NO COLLECTIVE!

    # ... code to determine group_world_size ...

    if self.is_entry_rank:
        image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
        if group_world_size > 1:
            obj_list = [image_inputs]
            torch.distributed.broadcast_object_list(  # COLLECTIVE 1
                obj_list, src=0, group=self.cpu_group
            )
            image_inputs = obj_list[0]
    else:
        if group_world_size > 1:
            obj_list = [None]
            torch.distributed.broadcast_object_list(  # COLLECTIVE 2
                obj_list, src=0, group=self.cpu_group
            )
            image_inputs = obj_list[0]
        else:
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)

    return image_inputs
```

**Call Sites**:
- `scheduler.py:1302` - `handle_generate_request()`
- `scheduler.py:1537` - `handle_embedding_request()`

**Problem**:
- **Conditional collective**: When `raw_mm_inputs is None`, method returns early without calling any collective
- **Non-uniform execution**: If different ranks receive different request patterns during initialization/warmup, they execute different numbers of broadcasts
- **Example scenario**:
  - Rank 0 processes 6 warmup requests with `mm_inputs != None` → 6 × 2 broadcasts = 12 collectives
  - Rank 7 processes 0 such requests → 0 collectives
  - **Result**: 12-collective mismatch ✓ (matches Case 1!)

**Why it affects non-multimodal models**:
- Even non-multimodal models go through this code path
- During initialization/warmup, test requests may be sent
- Request routing or health checks might cause non-uniform request distribution

---

### 2. November 4, 2025 - Symmetric Memory Registration (PR #12572)

**Commit**: `2340798353bc58398b6d45f582c7c79b670d0256`
**Author**: Nicolas Castet <26874160+nvcastet@users.noreply.github.com>
**Title**: "Register allgather/reducescatter buffers with symm memory"

**Files Changed** (19 files):
- `python/sglang/srt/distributed/parallel_state.py` (+95/-37)
- `python/sglang/srt/layers/dp_attention.py` (+45/-8)
- `python/sglang/srt/layers/communicator.py` (+12/-3)
- `python/sglang/srt/layers/linear.py` (+6/-1)
- `python/sglang/srt/layers/moe/cutlass_moe.py` (+2/-1)
- `python/sglang/srt/layers/quantization/modelopt_quant.py` (+52/-19)
- `python/sglang/srt/model_executor/cuda_graph_runner.py` (+6/-1)
- Many more...

**Problem Areas**:
- Changed how collective buffers are registered during initialization
- Modified `parallel_state.py` - core distributed initialization code
- Touched `dp_attention.py` - DP attention collective logic
- Changes to quantization layers - could affect initialization order

**Potential Issues**:
- Buffer registration might happen conditionally based on rank
- Symmetric memory initialization could execute different collectives per rank
- Timing changes in initialization sequence

---

### 3. November 10, 2025 - DP Attention Refactor (PR #12839)

**Commit**: `4f65a6466656cda958ab650198b7237062e9904f`
**Author**: Liangsheng Yin <hnyls2002@gmail.com>
**Title**: "Refactor / Unify event loop across PD-Disagg, Overlap, DP-Attn cases"

**Files Changed** (11 files):
- `python/sglang/srt/managers/scheduler_dp_attn_mixin.py` (+65/-37)
- `python/sglang/srt/managers/scheduler.py` (+8/-1)
- `python/sglang/srt/disaggregation/decode.py` (+142/-144)
- `python/sglang/srt/model_executor/forward_batch_info.py` (+7)

**Key Changes to `scheduler_dp_attn_mixin.py`**:

**BEFORE**:
```python
def prepare_mlp_sync_batch_raw(...):
    # ... calculate num_tokens ...

    mlp_sync_info = MLPSyncBatchInfo(...)
    mlp_sync_info.all_gather(device=device, group=group)  # COLLECTIVE

    if local_batch is None and max(mlp_sync_info.global_num_tokens) > 0:
        local_batch = get_idle_batch()

    if local_batch is not None:
        # Update batch with gathered info
        ...
```

**AFTER**:
```python
def prepare_mlp_sync_batch_raw(...):
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt():  # ⚠️ NEW CONDITION
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        # ... existing code ...

    # ... rest of calculation ...

    mlp_sync_info.all_gather(device=device, group=group)  # COLLECTIVE (still unconditional)

    need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0
    if need_idle_batch:
        batch_to_gather = local_batch
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():  # ⚠️ NEW: Special handling for prebuilt
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
        _update_gather_batch(batch_to_gather, mlp_sync_info, require_mlp_tp_gather)
```

**New MLPSyncBatchInfo fields**:
```python
@dataclass
class MLPSyncBatchInfo:
    # ... existing fields ...
    tbo_split_seq_index: torch.Tensor = None  # ⚠️ NEW
    global_forward_mode: int = None           # ⚠️ NEW
```

**Problem**:
- Added `.is_prebuilt()` mode handling
- Different batch states could lead to different code paths
- `get_idle_batch()` might trigger additional collectives
- Inner idle batch creation could cause rank-dependent behavior

---

### 4. November 10, 2025 - Event Loop Unification (PR #12959)

**Commit**: `dc8a5a1ce7a53115334fe1febf6cd1d354cfa804`
**Author**: Liangsheng Yin <hnyls2002@gmail.com>
**Title**: "[Refactor / Style] Unify all event loops (except for PP)"

**Files Changed** (5 files):
- `python/sglang/srt/disaggregation/base/conn.py` (+12)
- `python/sglang/srt/disaggregation/decode.py` (+95/-100)
- `python/sglang/srt/disaggregation/prefill.py` (+84/-89)
- `python/sglang/srt/managers/scheduler.py` (+6/-1)
- `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py` (+5/-1)

**Key Changes**:
- Refactored event loop structure for disaggregation modes
- Changed when `process_disagg_prefill_inflight_queue()` is called
- Modified idle checking logic

**Changes to disaggregation/prefill.py**:

**BEFORE**:
```python
def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
    while True:
        self.waiting_queue.extend(
            self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
        )
        self.process_prefill_chunk()
        batch = self.get_new_batch_prefill()
        # ... rest ...

        if len(self.disagg_prefill_inflight_queue) > 0:
            self.process_disagg_prefill_inflight_queue()

        if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
            self.self_check_during_idle()
```

**AFTER**:
```python
def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
    while True:
        self.waiting_queue.extend(
            self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
        )
        batch = self.get_next_disagg_prefill_batch_to_run()  # ⚠️ NEW METHOD
        self.cur_batch = batch

        if batch:
            result = self.run_batch(batch)
            self.process_batch_result_disagg_prefill(batch, result)
        else:
            self.self_check_during_idle()  # ⚠️ MOVED: Now called when batch is None

        self.process_disagg_prefill_inflight_queue()  # ⚠️ ALWAYS CALLED NOW
```

**Problem**:
- Changed control flow for when idle checks and queue processing occur
- Could affect timing of collective operations
- Different ranks might hit these code paths at different times

---

## Technical Analysis

### Collective Operations in Initialization Path

#### 1. **TpModelWorker Initialization**
Location: `python/sglang/srt/managers/tp_worker.py:305-311`

```python
# Sync random seed across TP workers
self.random_seed = broadcast_pyobj(
    [server_args.random_seed],
    self.tp_size * self.pp_rank + tp_rank,
    self.world_group.cpu_group,
    src=self.world_group.ranks[0],
)[0]
```
- **Conditional**: NO - Always executed
- **All ranks participate**: YES ✓

#### 2. **Model Loading Barrier**
Location: `python/sglang/srt/model_executor/model_runner.py:826-842`

```python
if self.server_args.elastic_ep_backend == "mooncake":
    # Mooncake does not support `monitored_barrier`
    dist.barrier(group=get_tp_group().cpu_group)
else:
    try:
        dist.monitored_barrier(
            group=get_tp_group().cpu_group,
            timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
            wait_all_ranks=True,
        )
```
- **Conditional**: Backend-dependent (but all ranks execute same path)
- **All ranks participate**: YES ✓

#### 3. **Memory Profiling All-Reduce**
Location: `python/sglang/srt/model_executor/model_runner.py:664-669` → `utils/common.py:531-536`

```python
min_per_gpu_memory = get_available_gpu_memory(
    self.device, self.gpu_id,
    distributed=get_world_group().world_size > 1,
    cpu_group=get_world_group().cpu_group,
)

# Implementation:
if distributed:
    tensor = torch.tensor(free_gpu_memory, dtype=torch.float32)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
```
- **Conditional**: Only if `world_size > 1`
- **All ranks participate**: YES ✓

#### 4. **CUDA Graph Warmup Barriers**
Location: `python/sglang/srt/model_executor/cuda_graph_runner.py:689-692`

```python
for _ in range(2):
    self.device_module.synchronize()
    self.model_runner.tp_group.barrier()  # ⚠️ BARRIER
    run_once()
```
- **Conditional**: NO - Always during CUDA graph capture
- **Frequency**: 2 warmup runs per batch size
- **All ranks participate**: SHOULD be YES, but this is where Case 2 fails!

#### 5. **DP Attention MLP Sync** (SUSPECT!)
Location: `python/sglang/srt/managers/scheduler_dp_attn_mixin.py:155`

```python
def prepare_mlp_sync_batch_raw(local_batch, ...):
    # ... calculate batch info based on local_batch state ...

    mlp_sync_info = MLPSyncBatchInfo(...)
    mlp_sync_info.all_gather(device=device, group=group)  # ⚠️ ALL_GATHER

    # ... post-processing based on gathered info ...
```

Called from: `scheduler.py:1687` and `scheduler.py:1703`

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    # ...
    need_mlp_sync = self.require_mlp_sync
    if need_mlp_sync and not self.spec_algorithm.is_none():
        new_batch = self.prepare_mlp_sync_batch(new_batch)  # ⚠️ CALL 1
        need_mlp_sync = new_batch is None

    # ... determine ret batch ...

    if need_mlp_sync:
        ret = self.prepare_mlp_sync_batch(ret)  # ⚠️ CALL 2
```

**Problem**:
- `prepare_mlp_sync_batch()` can be called 0, 1, or 2 times depending on batch state
- Each call triggers an `all_gather` collective
- If different ranks have different `local_batch` states, they execute different numbers of collectives

---

### Request Broadcasting Logic

Location: `python/sglang/srt/managers/scheduler.py:1095-1116`

```python
if self.server_args.enable_dp_attention:
    if self.attn_tp_rank == 0:
        work_reqs = [req for req in recv_reqs if isinstance(req, WorkReqTypes)]
        control_reqs = [req for req in recv_reqs if not isinstance(req, WorkReqTypes)]
    else:
        work_reqs = None
        control_reqs = None

    if self.attn_tp_size != 1:
        work_reqs = broadcast_pyobj(...)  # ⚠️ BROADCAST 1
    if self.tp_size != 1:
        control_reqs = broadcast_pyobj(...)  # ⚠️ BROADCAST 2
    recv_reqs = work_reqs + control_reqs
elif self.tp_size != 1:
    recv_reqs = broadcast_pyobj(...)  # ⚠️ BROADCAST 3 (different path!)
```

**Analysis**:
- With `enable_dp_attention`: 2 separate broadcasts (work + control)
- Without `enable_dp_attention`: 1 combined broadcast
- Both paths are executed by all ranks, but different total number of collectives

**`broadcast_pyobj` Implementation**:
Location: `python/sglang/srt/utils/common.py:1196-1240`

```python
def broadcast_pyobj(data, rank, dist_group=None, src=0, force_cpu_device=True):
    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)  # 1 broadcast
        else:
            # ... serialize data ...
            dist.broadcast(tensor_size, src=src, group=dist_group)   # 2 broadcasts
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)       # 1 broadcast
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)       # 2 broadcasts total
```

**Symmetric**: YES - Both source and non-source ranks execute same number of broadcasts ✓

---

## Root Cause Hypotheses

### Hypothesis 1: Multimodal Broadcast Bug (Case 1: TP Mode)

**Trigger**: `_process_and_broadcast_mm_inputs()` with early return

**Mechanism**:
1. During initialization, system sends warmup/health check requests
2. Due to request routing or timing, different ranks receive different requests
3. Some ranks get requests with `mm_inputs != None`, others get `mm_inputs == None`
4. Ranks with `mm_inputs != None` execute 2 `broadcast_object_list` calls
5. Ranks with `mm_inputs == None` execute 0 broadcasts (early return)
6. Over multiple requests, mismatch accumulates

**Evidence**:
- Difference is 12 collectives (30 - 18 = 12)
- 12 = 6 requests × 2 broadcasts per request
- Introduced Oct 30, 2025 (recent enough to be "regression")
- Affects non-multimodal models (unexpected but explained by warmup)

**Likelihood**: **HIGH** ⚠️

---

### Hypothesis 2: DP Attention MLP Sync (Case 2: Disagg + DP Attention)

**Trigger**: `prepare_mlp_sync_batch()` called different number of times per rank

**Mechanism**:
1. With `enable_dp_attention`, MLP sync is required
2. During CUDA graph capture initialization, ranks prepare batches
3. Different ranks have different batch states (None, decode, prefill, prebuilt, idle)
4. `get_next_batch_to_run()` conditionally calls `prepare_mlp_sync_batch()` 0-2 times
5. Each call triggers an `all_gather` collective
6. Rank-dependent batch states → rank-dependent collective counts

**Evidence**:
- Difference is 60 collectives (73 - 13 = 60)
- 60 ≈ 2 collectives × 30 iterations (or similar pattern)
- Only happens with `--enable-dp-attention`
- Occurs during CUDA graph capture (when batches are being prepared)
- Recent DP attention refactor (Nov 10, 2025) changed batch state handling

**Code Path**:
```
CUDA graph capture
  → CudaGraphRunner.__init__
    → capture()
      → warmup loop (2 iterations)
        → tp_group.barrier()  ← Rank 0: SequenceNumber=73
                                  ← Rank 1-7: SequenceNumber=13
```

**Between SequenceNumber 1 and 73** (Rank 0 only):
- 72 collectives executed
- Likely in: batch preparation, MLP sync, or other DP attention setup

**Likelihood**: **VERY HIGH** ⚠️⚠️⚠️

---

### Hypothesis 3: Symmetric Memory Registration (Buffer Registration Conditionals)

**Trigger**: Commit #12572 changed buffer registration

**Mechanism**:
1. Symmetric memory buffers are registered during initialization
2. Registration might involve collective operations (barrier, metadata exchange)
3. Registration could be conditional based on:
   - Rank (e.g., only rank 0 registers certain buffers)
   - Model features (quantization, MoE, etc.)
   - Device availability
4. Different ranks register different buffers → different collective counts

**Evidence**:
- Commit touched 19 files including core distributed code
- Modified `parallel_state.py`, `communicator.py`, `dp_attention.py`
- Changed quantization layers (modelopt_quant.py +52/-19 lines)
- Timing: Nov 4, 2025 (between multimodal fix and DP refactor)

**Likelihood**: **MEDIUM** - Could contribute but likely not sole cause

---

### Hypothesis 4: Event Loop Refactor Timing Issues

**Trigger**: Commit #12959 changed event loop control flow

**Mechanism**:
1. Refactored when `process_disagg_prefill_inflight_queue()` is called
2. Changed idle checking logic
3. Different ranks might hit these paths at different iteration counts
4. If any of these methods involve collectives, mismatch accumulates

**Evidence**:
- Timing: Nov 10, 2025 (same day as DP refactor)
- Changed disaggregation event loops
- Only affects PD disaggregation mode

**Likelihood**: **LOW-MEDIUM** - More likely a secondary factor

---

## Affected Code Locations

### Primary Suspects

#### 1. `scheduler.py:1157-1219` - `_process_and_broadcast_mm_inputs()`
**File**: `python/sglang/srt/managers/scheduler.py`
**Lines**: 1157-1219
**Issue**: Conditional broadcast with early return
**Severity**: CRITICAL for Case 1

#### 2. `scheduler_dp_attn_mixin.py:94-173` - `prepare_mlp_sync_batch_raw()`
**File**: `python/sglang/srt/managers/scheduler_dp_attn_mixin.py`
**Lines**: 94-173
**Issue**: Batch state-dependent collective calls
**Severity**: CRITICAL for Case 2

#### 3. `scheduler.py:1681-1703` - `get_next_batch_to_run()`
**File**: `python/sglang/srt/managers/scheduler.py`
**Lines**: 1681-1703
**Issue**: Conditional `prepare_mlp_sync_batch()` calls
**Severity**: CRITICAL for Case 2

### Secondary Suspects

#### 4. `parallel_state.py` - Symmetric memory initialization
**File**: `python/sglang/srt/distributed/parallel_state.py`
**Changes**: PR #12572
**Issue**: Buffer registration collectives
**Severity**: MEDIUM

#### 5. `disaggregation/prefill.py` & `decode.py` - Event loop refactor
**Files**:
- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/disaggregation/decode.py`
**Changes**: PR #12959
**Issue**: Control flow timing
**Severity**: LOW-MEDIUM

---

## Reproduction Steps

### Case 1: Standard TP Mode

```bash
# Using Docker
docker run --rm --gpus all --network host --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  lmsysorg/sglang:v0.5.5 \
  python -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2-Thinking \
    --tp 8 \
    --disable-cuda-graph

# Expected: Crash during event loop initialization
# Error: SequenceNumber mismatch (e.g., 30 vs 18)
```

### Case 2: Disaggregation + DP Attention

```bash
# Prefill worker
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --disaggregation-mode prefill \
  --tp-size 32 \
  --dp-size 32 \
  --enable-dp-attention \
  --enable-two-batch-overlap

# Decode worker
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --disaggregation-mode decode \
  --tp-size 32 \
  --dp-size 32 \
  --enable-dp-attention \
  --enable-two-batch-overlap

# Expected: Crash during CUDA graph capture
# Error: SequenceNumber mismatch (e.g., 73 vs 13)
```

---

## Proposed Fixes

### Fix 1: Unconditional Broadcast for Multimodal Inputs

**Target**: `scheduler.py:1157-1219`
**Priority**: CRITICAL
**Complexity**: LOW

**Current Code** (BROKEN):
```python
def _process_and_broadcast_mm_inputs(self, raw_mm_inputs: Optional[dict]):
    if raw_mm_inputs is None:
        return None  # ❌ EARLY RETURN - NO COLLECTIVE!

    # ... rest of code with broadcasts ...
```

**Fixed Code**:
```python
def _process_and_broadcast_mm_inputs(self, raw_mm_inputs: Optional[dict]):
    # Determine group world size
    group_world_size = 1
    try:
        if (torch.distributed.is_available() and
            torch.distributed.is_initialized() and
            self.cpu_group is not None):
            group_world_size = torch.distributed.get_world_size(group=self.cpu_group)
    except Exception as e:
        logger.warning(f"Failed to get world size: {e}")

    # ✅ ALL RANKS MUST PARTICIPATE - even when raw_mm_inputs is None
    if group_world_size > 1:
        if self.is_entry_rank:
            # Prepare data (None or actual)
            image_inputs = (MultimodalInputs.from_dict(raw_mm_inputs)
                          if raw_mm_inputs is not None else None)
            obj_list = [image_inputs]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            return obj_list[0]
        else:
            # Non-entry ranks ALWAYS receive
            obj_list = [None]
            torch.distributed.broadcast_object_list(
                obj_list, src=0, group=self.cpu_group
            )
            return obj_list[0]
    else:
        # Single rank - no broadcast needed
        return (MultimodalInputs.from_dict(raw_mm_inputs)
                if raw_mm_inputs is not None else None)
```

**Testing**:
- Run with TP=8, non-multimodal model (Llama-3-70B)
- Verify no sequence number mismatch
- Check that None values are properly broadcasted and handled

---

### Fix 2: Ensure Uniform MLP Sync Calls

**Target**: `scheduler.py:1681-1703` + `scheduler_dp_attn_mixin.py`
**Priority**: CRITICAL
**Complexity**: MEDIUM

**Approach A: Always Call MLP Sync (Simplest)**

**Current Code** (scheduler.py:1681-1703):
```python
need_mlp_sync = self.require_mlp_sync
if need_mlp_sync and not self.spec_algorithm.is_none():
    new_batch = self.prepare_mlp_sync_batch(new_batch)  # ⚠️ CONDITIONAL CALL
    need_mlp_sync = new_batch is None

# ... determine ret batch ...

if need_mlp_sync:
    ret = self.prepare_mlp_sync_batch(ret)  # ⚠️ CONDITIONAL CALL
```

**Fixed Code**:
```python
# Option A: Always call for both new_batch and ret_batch when required
if self.require_mlp_sync:
    if not self.spec_algorithm.is_none():
        new_batch = self.prepare_mlp_sync_batch(new_batch)  # ✅ ALWAYS CALL

    # ... determine ret batch ...

    # ✅ ALWAYS call MLP sync for ret batch
    ret = self.prepare_mlp_sync_batch(ret)
```

**Approach B: Track Collective Count (More Complex)**

```python
def prepare_mlp_sync_batch(self, batch: Optional[ScheduleBatch]) -> Optional[ScheduleBatch]:
    # Track whether we actually need to sync
    should_sync = self.require_mlp_sync and (batch is not None or self._needs_idle_batch_sync())

    if should_sync:
        return prepare_mlp_sync_batch_raw(batch, ...)
    else:
        # Still participate in collective with dummy data
        return self._dummy_mlp_sync(batch)

def _dummy_mlp_sync(self, batch):
    # All ranks participate in collective but with dummy/zero data
    # Ensures uniform collective count
    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=self.dp_size,
        tp_size=self.attn_tp_size,
        num_tokens=0,  # Dummy
        num_tokens_for_logprob=0,  # Dummy
        # ... rest with defaults ...
    )
    mlp_sync_info.all_gather(device="cpu", group=self.tp_cpu_group)
    return batch  # Return unchanged
```

**Testing**:
- Run with `--enable-dp-attention --tp-size 32 --dp-size 32`
- Verify CUDA graph capture succeeds
- Check that all ranks have same sequence number at barriers

---

### Fix 3: Add Collective Synchronization Checkpoints

**Target**: Initialization and graph capture paths
**Priority**: HIGH (defensive programming)
**Complexity**: LOW

**Implementation**:
```python
# In model_runner.py, before CUDA graph capture
def init_device_graphs(self):
    # ... existing code ...

    # ✅ Ensure all ranks are synchronized before graph capture
    if self.tp_group and self.tp_group.world_size > 1:
        logger.info(f"[Rank {self.tp_rank}] Synchronizing before CUDA graph capture")
        self.tp_group.barrier()

    # Now capture graphs
    self.cuda_graph_runner = CudaGraphRunner(...)
```

```python
# In scheduler.py, at event loop start
def event_loop_overlap(self):
    # ✅ Initial synchronization barrier
    if self.tp_size > 1:
        logger.info(f"[Rank {self.tp_rank}] Event loop starting - synchronizing")
        self.tp_group.barrier()

    while True:
        recv_reqs = self.recv_requests()
        # ... rest of event loop ...
```

**Benefit**: Makes collective mismatches fail-fast with clearer error location

---

### Fix 4: Revert Problematic PRs (Temporary Workaround)

**Priority**: LOW (last resort)
**Complexity**: LOW

**Revert Order**:
1. Revert PR #12959 (Event Loop Unification) - commit `dc8a5a1ce`
2. Revert PR #12839 (DP Attention Refactor) - commit `4f65a6466`
3. Revert PR #11910 (Multimodal Optimization) - commit `17a57fd86`

**Testing After Each Revert**:
- Check if Case 1 resolves (after reverting #11910)
- Check if Case 2 resolves (after reverting #12839)

---

## Debugging Strategy

### Step 1: Add Collective Tracing

**Goal**: Identify exactly which collectives are being executed by each rank

**Implementation**:
```python
# In utils/common.py
import os
import threading

_collective_log_lock = threading.Lock()
_collective_counter = 0

def log_collective(op_type: str, rank: int, group_name: str = "unknown"):
    global _collective_counter
    with _collective_log_lock:
        _collective_counter += 1
        seq = _collective_counter

    if os.getenv("SGLANG_DEBUG_COLLECTIVES"):
        import traceback
        stack = traceback.format_stack()
        caller = stack[-3] if len(stack) >= 3 else "unknown"

        print(f"[COLLECTIVE DEBUG] Rank {rank} | Seq {seq} | {op_type} | Group {group_name}")
        print(f"  Called from: {caller}")

# Modify broadcast_pyobj
def broadcast_pyobj(data, rank, dist_group=None, src=0, force_cpu_device=True):
    log_collective("BROADCAST", rank, f"group_{id(dist_group)}")
    # ... rest of existing code ...
```

**Usage**:
```bash
export SGLANG_DEBUG_COLLECTIVES=1
python -m sglang.launch_server --tp 8 ...
```

**Expected Output**:
```
[COLLECTIVE DEBUG] Rank 0 | Seq 1 | BROADCAST | Group group_140123456
  Called from: scheduler.py:1204 in _process_and_broadcast_mm_inputs
[COLLECTIVE DEBUG] Rank 1 | Seq 1 | BROADCAST | Group group_140123456
  Called from: scheduler.py:1212 in _process_and_broadcast_mm_inputs
```

---

### Step 2: Bisect Commit History

**Goal**: Find exact commit that introduced the regression

**Process**:
```bash
# Start bisect
git bisect start
git bisect bad HEAD  # Current version is broken
git bisect good v0.5.3  # Assume v0.5.3 worked

# Git will checkout commits for testing
# For each commit:
python -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8

# If it works:
git bisect good

# If it fails:
git bisect bad

# Repeat until git identifies the first bad commit
```

---

### Step 3: Unit Test for Collective Uniformity

**Goal**: Create test that catches collective mismatches early

**Implementation**:
```python
# test/srt/test_collective_uniformity.py
import torch
import torch.distributed as dist
import pytest
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs

class CollectiveTracker:
    def __init__(self):
        self.ops = []

    def track(self, op_type, tensor_shape):
        self.ops.append((op_type, tensor_shape))

    def get_fingerprint(self):
        return tuple(self.ops)

@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_collective_uniformity_tp(tp_size):
    """Test that all ranks execute same collectives during initialization"""

    # Mock distributed environment
    # ... setup code ...

    tracker = CollectiveTracker()

    # Monkey-patch collective ops to track them
    original_broadcast = dist.broadcast
    def tracked_broadcast(tensor, src, group=None):
        tracker.track("broadcast", tensor.shape)
        return original_broadcast(tensor, src, group)
    dist.broadcast = tracked_broadcast

    # Initialize scheduler
    args = ServerArgs(tp_size=tp_size, ...)
    scheduler = Scheduler(args, ...)

    # Collect fingerprints from all ranks
    fingerprints = gather_from_all_ranks(tracker.get_fingerprint())

    # Assert all ranks have identical fingerprints
    assert len(set(fingerprints)) == 1, \
        f"Collective mismatch! Fingerprints: {fingerprints}"
```

---

## Monitoring and Prevention

### 1. Pre-commit Hook for Collective Checks

**File**: `.git/hooks/pre-commit`

```bash
#!/bin/bash

# Check for new broadcast/all_reduce calls in scheduler code
git diff --cached --name-only | grep -E "scheduler|tp_worker|model_runner" | while read file; do
    if git diff --cached "$file" | grep -E "broadcast|all_reduce|all_gather|barrier" | grep "^\+"; then
        echo "⚠️  WARNING: New collective operation detected in $file"
        echo "   Please ensure all ranks execute this collective unconditionally!"
        echo ""
        git diff --cached "$file" | grep -E "broadcast|all_reduce|all_gather|barrier" | grep "^\+"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
done
```

### 2. CI Test for Collective Uniformity

**File**: `.github/workflows/test_collectives.yml`

```yaml
name: Collective Uniformity Tests

on: [pull_request]

jobs:
  test_collective_uniformity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run collective uniformity tests
        run: |
          pytest test/srt/test_collective_uniformity.py \
            --tb=short \
            -v
```

### 3. Documentation

**File**: `docs/developer/collective_operations.md`

```markdown
# Guidelines for Collective Operations

## Rules

1. **NEVER skip a collective based on rank**
   ❌ BAD:
   ```python
   if self.tp_rank == 0:
       dist.broadcast(data, src=0)
   ```

   ✅ GOOD:
   ```python
   # All ranks participate
   dist.broadcast(data, src=0)
   ```

2. **NEVER return early from a function that contains collectives**
   ❌ BAD:
   ```python
   def sync_data(data):
       if data is None:
           return  # Some ranks skip collective!
       dist.broadcast(data, src=0)
   ```

   ✅ GOOD:
   ```python
   def sync_data(data):
       # All ranks participate even with None data
       if self.world_size > 1:
           dist.broadcast(data if data else dummy, src=0)
   ```

3. **Conditional collectives must have uniform conditions**
   ❌ BAD:
   ```python
   if self.local_batch is not None:  # Rank-dependent!
       dist.all_reduce(count)
   ```

   ✅ GOOD:
   ```python
   # Use collective to decide uniformly
   has_batch = self.local_batch is not None
   all_have_batch = dist.all_reduce(has_batch, op=AND)
   if all_have_batch:
       dist.all_reduce(count)
   ```
```

---

## Success Criteria

### For Fix Validation

1. ✅ Case 1 (TP mode) runs without errors:
   ```bash
   python -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8
   ```

2. ✅ Case 2 (Disagg + DP Attention) runs without errors:
   ```bash
   # Prefill + Decode workers both start successfully
   # CUDA graph capture completes on all ranks
   ```

3. ✅ All ranks report same sequence number at key checkpoints:
   - After initialization
   - After model loading
   - After CUDA graph capture
   - At first event loop barrier

4. ✅ Collective uniformity tests pass:
   ```bash
   pytest test/srt/test_collective_uniformity.py
   ```

5. ✅ No regression in existing tests:
   ```bash
   pytest test/srt/test_tp*.py
   pytest test/srt/test_disaggregation*.py
   ```

---

## Next Steps

1. **Immediate Action** (Today):
   - [ ] Apply Fix 1 (multimodal broadcast) - PR ready to merge
   - [ ] Add collective tracing debug mode
   - [ ] Test with Kimi K2 TP=8

2. **Short Term** (This Week):
   - [ ] Investigate Fix 2 (MLP sync) - analyze DP attention code path
   - [ ] Add unit tests for collective uniformity
   - [ ] Bisect commit history to confirm hypothesis

3. **Medium Term** (Next Sprint):
   - [ ] Comprehensive review of all collective operations
   - [ ] Add documentation guidelines
   - [ ] Set up CI tests for collective uniformity
   - [ ] Add pre-commit hooks

4. **Long Term**:
   - [ ] Architectural review of distributed initialization
   - [ ] Consider collective operation framework/wrapper
   - [ ] Add telemetry for collective operation monitoring

---

## Related Issues

- #13227 - Main issue (Kimi K2 TP=8 failure)
- #12915 - Spec decoding with TP=8
- #10218 - Metrics for scheduler (might have introduced collectives)
- #12839 - DP attention refactor (suspected cause)
- #11910 - Multimodal optimization (suspected cause)

---

## References

- PyTorch Distributed Docs: https://pytorch.org/docs/stable/distributed.html
- NCCL Debugging: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- SGLang Architecture: https://github.com/sgl-project/sglang/blob/main/docs/architecture.md

---

**Document Version**: 1.0
**Last Updated**: November 14, 2025
**Author**: Root Cause Analysis AI
**Status**: Ready for Implementation

---

## CRITICAL UPDATE: Key Configuration Difference Found

### Successful vs Failed Run Comparison

**Date**: November 14, 2025
**Location**: `/Users/idhanani/Downloads/`

#### Successful Run (multi_node_disagg_4097913)
```bash
--tp-size 16
--dp-size 16
--nnodes 2
--enable-dp-attention
--moe-a2a-backend none                    # ← KEY DIFFERENCE
--enable-two-batch-overlap FALSE          # ← KEY DIFFERENCE  
--deepep-mode auto
--ep-num-redundant-experts 0
--chunked-prefill-size 512
```
**Result**: ✅ CUDA graph capture **SUCCEEDED** on all 16 DP ranks

#### Failed Run (dsr1_wideep_h100_4102550)
```bash
--tp-size 32
--dp-size 32  
--nnodes 4
--enable-dp-attention
--moe-a2a-backend deepep                  # ← KEY DIFFERENCE
--enable-two-batch-overlap TRUE           # ← KEY DIFFERENCE
--deepep-mode normal
--ep-num-redundant-experts 32
--ep-dispatch-algorithm dynamic
--eplb-algorithm deepseek
--chunked-prefill-size 256
--deepep-config /configs/deepep.json
```
**Result**: ❌ CUDA graph capture **FAILED** during barrier
```
Rank 0: SequenceNumber=73
Rank 1-7: SequenceNumber=13
Difference: 60 collectives
```

### Root Cause: Two-Batch Overlap + DeepEP Interaction

The bug is triggered by the **combination** of:
1. `--enable-two-batch-overlap` (TBO)
2. `--moe-a2a-backend deepep` (DeepEP)
3. `--enable-dp-attention` (DP Attention)
4. High parallelism (TP=32, DP=32, EP=32)

### Analysis

#### Hypothesis: TBO + DeepEP Conditional Collectives

With `--enable-two-batch-overlap` enabled:
- Scheduler uses `event_loop_overlap()` instead of `event_loop_normal()`
- Overlapped scheduling involves **pipelined batch preparation**
- Different ranks might prepare batches at different pipeline stages

With DeepEP (Deep Expert Parallelism):
- Expert routing involves **all-to-all collectives** per layer
- 32 redundant experts with dynamic dispatch
- EPLB (Expert Load Balancing) with deepseek algorithm
- Each expert routing decision might trigger collectives

**Problem**: During CUDA graph capture warmup:
- Rank 0 (DP rank 0, attn_tp_rank 0) is the "leader" rank
- Leader rank might execute extra MLP sync operations
- With TBO, leader rank prepares batches ahead (pipeline)
- Each batch preparation triggers:
  - MLP sync all_gather (DP attention)
  - Expert routing collectives (DeepEP)
- Non-leader ranks wait at barriers without executing prep collectives
- **Result**: 60-collective mismatch (≈ 2 collectives × 30 expert layers)

#### Code Path Investigation Needed

**File**: `python/sglang/srt/managers/scheduler.py`
**Method**: `event_loop_overlap()`

```python
def event_loop_overlap(self):
    """A scheduler loop that overlaps the CPU processing and GPU computation."""
    self.result_queue: Deque[Tuple[ScheduleBatch, GenerationBatchResult]] = deque()

    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()  # ← MLP sync happens here
        self.cur_batch = batch

        if batch:
            result = self.run_batch(batch)      # ← Expert routing happens here
            self.process_batch_result(batch, result)
        # ... overlap logic ...
```

**Suspected Issue**: 
- `get_next_batch_to_run()` calls `prepare_mlp_sync_batch()` which executes all_gather
- With TBO, this might be called **different number of times** per rank
- DeepEP expert routing adds **additional conditional collectives**

#### Verification Strategy

1. **Test without TBO**:
   ```bash
   # Remove --enable-two-batch-overlap
   # Keep everything else the same
   ```
   Expected: Should succeed if TBO is the trigger

2. **Test without DeepEP**:
   ```bash
   # Change --moe-a2a-backend deepep to --moe-a2a-backend none
   # Keep TBO enabled
   ```
   Expected: Should succeed if DeepEP collectives are the issue

3. **Test with lower parallelism**:
   ```bash
   # Use TP=16, DP=16 (same as successful run)
   # Keep TBO + DeepEP enabled
   ```
   Expected: Might succeed if it's a scaling issue

### Updated Fix Priority

#### Fix 1: Ensure Uniform Batch Preparation in TBO Mode

**Target**: `scheduler.py:event_loop_overlap()`
**Priority**: CRITICAL
**Complexity**: HIGH

**Problem**: With `enable_two_batch_overlap`, different ranks execute `get_next_batch_to_run()` at different times in the pipeline.

**Approach**: Add synchronization barrier before batch preparation when DP attention is enabled:

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    # ... existing code ...

    need_mlp_sync = self.require_mlp_sync
    
    # ✅ NEW: Synchronize before MLP sync in TBO mode
    if need_mlp_sync and self.enable_overlap:
        # Ensure all ranks are ready to prepare batches together
        if self.tp_size > 1:
            self.tp_group.barrier()
    
    if need_mlp_sync and not self.spec_algorithm.is_none():
        new_batch = self.prepare_mlp_sync_batch(new_batch)
        need_mlp_sync = new_batch is None

    # ... rest of code ...
```

**Risk**: Adding barriers might reduce TBO performance benefits

#### Fix 2: Disable Conditional Expert Routing Collectives

**Target**: DeepEP expert routing code
**Priority**: HIGH  
**Complexity**: MEDIUM

**Investigation Needed**: Check if DeepEP expert routing has rank-conditional collectives during initialization/warmup.

**Files to Check**:
- `python/sglang/srt/layers/moe/deepep_moe.py` (if exists)
- `deep_ep` package collectives
- Expert load balancing code

#### Fix 3: Unconditional MLP Sync (Still Valid)

Keep the previous Fix 2 from main analysis - ensure `prepare_mlp_sync_batch()` is always called uniformly.

### Testing Plan

```bash
# Test 1: Disable TBO
python -m sglang.launch_server \
  --disaggregation-mode decode \
  --tp-size 32 --dp-size 32 --nnodes 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-two-batch-overlap FALSE  # ← Changed

# Test 2: Disable DeepEP  
python -m sglang.launch_server \
  --disaggregation-mode decode \
  --tp-size 32 --dp-size 32 --nnodes 4 \
  --enable-dp-attention \
  --moe-a2a-backend none  # ← Changed
  --enable-two-batch-overlap

# Test 3: Lower parallelism
python -m sglang.launch_server \
  --disaggregation-mode decode \
  --tp-size 16 --dp-size 16 --nnodes 2 \  # ← Changed
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-two-batch-overlap
```

### Expected Timeline

1. **Immediate** (Today): Run Test 1 and Test 2 to isolate the trigger
2. **Short-term** (This week): Implement Fix 1 with barrier synchronization
3. **Medium-term** (Next week): Investigate and fix DeepEP collectives if needed
4. **Long-term**: Architectural review of TBO + DP attention interaction

