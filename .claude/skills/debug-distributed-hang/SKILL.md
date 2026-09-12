---
name: debug-distributed-hang
description: Debug hanging issues in SGLang distributed inference (TP/PP/DP/EP). Covers identifying hang locations via py-spy/watchdog/cuda coredump, per-rank logging to find state divergence, binary-search methodology for locating the first diverge point, and fix patterns. Use when a multi-GPU SGLang run hangs, freezes, or times out during collective operations.
---

# Debugging Distributed Hangs in SGLang

## Overview

Hangs in distributed inference happen when ranks diverge in state, causing collective operations (AllGather, AllReduce, Broadcast, Barrier) to deadlock. Common causes:

- **Size mismatch**: ranks pass different tensor sizes to a collective
- **Branch divergence**: one rank enters a collective, another skips it
- **Cascading state drift**: a small non-determinism (e.g., floating-point) propagates into different batch structures
- **Resource exhaustion**: one rank OOMs or crashes, others wait forever

## Prerequisites

- **py-spy**: `pip install py-spy` or system package. Requires root or `CAP_SYS_PTRACE` to attach to running processes.
- **cuda-gdb**: Ships with the CUDA toolkit. Ensure it's on your `PATH`.

## Step 1: Confirm and Locate the Hang

### 1a. Watchdog / py-spy

SGLang's watchdog automatically dumps py-spy traces on timeout. Look for:

```
Scheduler watchdog timeout (self.watchdog_timeout=300, self.soft=False)
```

The py-spy dump shows the stack trace of each thread. The hanging thread is typically blocked in a CUDA synchronize or NCCL collective:

```
Thread (active): "MainThread"
    cuStreamSynchronize (libcuda.so)
    ...
    forward_extend (model_runner.py)
```

SGLang has two watchdog modes (see `python/sglang/srt/utils/watchdog.py`):
- **Hard watchdog** (`soft=False`, default): dumps py-spy traces then sends `SIGQUIT` to kill the parent process.
- **Soft watchdog** (`soft=True`): only logs the timeout without killing the process, giving you more time to manually attach debuggers or collect coredumps.

If the watchdog doesn't trigger, manually dump:

```bash
py-spy dump --pid <scheduler_pid>
```

### 1b. NCCL Debug Logging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

Look for the last collective logged before the hang. Mismatched sizes show up as one rank waiting and another never entering.

### 1c. CUDA Coredump

When a process hangs, you can trigger a GPU coredump on demand to see which kernel is stuck. Set these env vars before launching:

```bash
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_COREDUMP_PIPE="/tmp/cuda_pipe_%h_%p"
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h_%p"
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
```

While the process is hanging, find the pipe via `/proc/<pid>/fd/` and write to it to trigger the dump:

```bash
ls /proc/<pid>/fd/ -la 2>/dev/null | grep cuda_pipe
dd if=/dev/zero bs=1M count=1 > /tmp/cuda_pipe_<hostname>_<pid>
```

Alternatively, if you don't need to keep the process alive, `kill -SIGABRT <pid>` also triggers a CUDA coredump (but terminates the process).

Then open with `cuda-gdb --batch -ex "target cudacore <coredump_file>"`. On load, it immediately shows which kernel is stuck. For example:

```
Opening GPU coredump: <coredump_file>
[Current focus set to CUDA kernel 0, grid 622721, cluster (4,0,0), block (16,0,0), thread (64,0,0), device 0, sm 0, warp 0, lane 0]
#0  0x00007f8029b2b040 in ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)<<<(24,1,1),(512,1,1)>>> ()
```

This told us the hang was in an NCCL AllGather — not a compute kernel. Combined with the py-spy stack pointing to `LogitsProcessor.forward` → `tensor_model_parallel_all_gather`, we knew it was an AllGather size mismatch between TP ranks.


### 1d. Identify the Collective

From the stack traces and logs, identify:
- Which collective hangs (AllGather, AllReduce, Broadcast)
- Which code path invokes it (e.g., `LogitsProcessor`, `tensor_model_parallel_all_gather`)
- Whether it's a size mismatch or a missing participant

## Step 2: Per-Rank Logging

The key technique: each rank writes its own log file so you can diff them.

### Setup Pattern

```python
import os

_debug_files = {}

def get_debug_file(rank):
    key = f"rank{rank}"
    if key not in _debug_files:
        _debug_files[key] = open(f"/tmp/debug_rank{rank}.log", "w")
    return _debug_files[key]
```

Gate logging behind an env var to avoid overhead in production. `SGLANG_DEBUG_HANG` is not a built-in SGLang env var — you need to add this check yourself in the code you're instrumenting:

```python
if os.environ.get("SGLANG_DEBUG_HANG"):
    f = get_debug_file(rank)
    f.write(f"EVENT_NAME key1={val1} key2={val2}\n")
    f.flush()
```

### What to Log

Log structured events at key state-mutation points:

```python
f.write(f"SCHED_BATCH step={step} num_reqs={n} extend_lens={lens}\n")
f.write(f"VERIFY predict_hash={hash} accept_len={alen}\n")
f.write(f"CACHE_INSERT rid={rid} num_tokens={n}\n")
```

Use consistent event names (uppercase prefix) for easy grep/diff.

### Hash Large Tensors

For tensor values, compute a hash instead of dumping raw data:

```python
import hashlib
h = hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()[:8]
f.write(f"LOGITS logits_hash={h}\n")
```

For token ID lists, `str(list).encode()` works:

```python
h = hashlib.md5(str(tensor.tolist()).encode()).hexdigest()[:8]
```

### Avoid Implicit Synchronization

`tensor.cpu()`, `tensor.tolist()`, and `tensor.numpy()` all trigger CUDA synchronization. This can:
- Change timing and mask or move the hang
- Deadlock if the log point is between two collectives that must run back-to-back

Prefer logging values that are already on CPU (e.g., Python ints, list lengths, request IDs). When you must hash a GPU tensor, do it at a point where the GPU is already idle (e.g., between scheduler steps, not inside a model forward pass).

## Step 3: Diff to Find the Diverge Point

### Basic Diff

```bash
# Extract specific event type
grep "^VERIFY" /tmp/debug_rank0.log > /tmp/v_r0.txt
grep "^VERIFY" /tmp/debug_rank1.log > /tmp/v_r1.txt
diff /tmp/v_r0.txt /tmp/v_r1.txt | head -20
```

### Count Events

```bash
grep -c "^VERIFY" /tmp/debug_rank*.log
```

If counts differ, one rank executed more iterations — that's already a diverge signal.

### Find First Diverge

The first diff line tells you the exact step where ranks diverge. All lines before it are identical — the root cause is at or before this step.

## Step 4: Binary-Search the Root Cause

Once you find the diverging event, trace backwards:

### 4a. Identify Inputs

For the diverging operation, list all its inputs. Add hash logging for each:

```python
f.write(
    f"OP_INPUTS input_a_hash={h_a} input_b_hash={h_b} "
    f"input_c_hash={h_c} input_d_hash={h_d}\n"
)
```

### 4b. Diff Inputs Across Ranks

Compare the hashes. Some inputs will match, some won't. The non-matching input is where divergence entered.

### 4c. Recurse

For the non-matching input, trace where it was produced and repeat: hash its inputs, diff across ranks, find the divergent one. Continue until you reach the root cause.

## Step 5: Common Root Causes and Fixes

### Floating-Point Non-Determinism

**Symptom**: All "logical" inputs are identical (same logits after all-gather), but derived floating-point values (softmax, probabilities) differ across GPUs.

**Example**: EAGLE speculative decoding — `F.softmax` → `top_k_renorm_prob` → `top_p_renorm_prob` produces slightly different `target_probs` on each GPU. The sampling kernel then picks different tokens. These flow into `output_ids` → radix cache → different prefix match depths → different `extend_seq_lens` → AllGather size mismatch → hang.

### Random Number Divergence

**Symptom**: Operations using `torch.rand` produce different values on each rank.

**Fix**: Generate on rank 0 and broadcast, or use a shared seed.

### Conditional Code Paths

**Symptom**: A condition (e.g., memory check, queue length) evaluates differently on different ranks, causing one rank to enter a collective while another skips it.

**Fix**: Synchronize the condition value before branching, or restructure to ensure all ranks take the same path.

### Pipeline Parallel (PP) Send/Recv Mismatch

**Symptom**: In PP setups, one stage issues a `send` that the next stage never `recv`s (or vice versa), causing both to block indefinitely. Unlike TP hangs (collective mismatches), PP hangs typically involve point-to-point operations.

**Fix**: Ensure all stages agree on the number of microbatches and the sequence of send/recv calls for each microbatch.

## Step 6: Verify the Fix

Run the failing test multiple times to confirm the fix is stable. Intermittent hangs require many runs. A test that hung ~30% of the time needs at least 10 clean passes to be confident.

## Quick Reference

| Technique | When to Use |
|-----------|-------------|
| py-spy dump | First step — see where each rank is stuck |
| `NCCL_DEBUG=INFO` | Identify which collective and sizes |
| CUDA coredump + `cuda-gdb` | See which GPU kernel is blocked |
| Per-rank log files | Compare rank states over time |
| Hash of tensors | Efficiently compare large tensors across ranks |
| `diff` on extracted events | Find the exact step of divergence |
| `broadcast(result, src=0)` | Fix floating-point or sampling non-determinism |
