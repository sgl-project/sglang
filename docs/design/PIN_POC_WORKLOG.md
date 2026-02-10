# PIN POC Worklog

**Branch**: `idhanani/dyn-1986-poc-pin`
**Worktree**: `/home/ubuntu/sglang-poc-pin`
**Linear**: DYN-1986

---

## Summary

Implemented `pin_blocks` / `unpin_blocks` at the RadixCache level in SGLang. Pinning a block sets `lock_ref > 0` on the corresponding TreeNode, which removes it from `evictable_leaves` and makes it immune to eviction. This reuses SGLang's existing eviction protection mechanism (same one used by active requests).

**Headline result**: PIN delivers **up to 11.1x TTFT speedup** at conversation depth 16. Baseline TTFT grows linearly with depth (421ms at depth 0 to 2745ms at depth 16); pinned TTFT stays flat at ~230-250ms regardless of depth.

---

## File Inventory

### Core Implementation (modified files)

| File | Changes |
|------|---------|
| `python/sglang/srt/mem_cache/radix_cache.py` | `block_hash_index` dict, `external_pin_count` dict, `_index_node_hashes()`, `_unindex_node_hashes()`, `pin_blocks()`, `unpin_blocks()`. Index maintenance in `_record_store_event` and `_record_remove_event`. Clears in `reset()`. |
| `python/sglang/srt/entrypoints/http_server.py` | `/hicache/pin_blocks` and `/hicache/unpin_blocks` HTTP endpoints |
| `python/sglang/srt/managers/io_struct.py` | `PinBlocksReqInput`, `UnpinBlocksReqInput` IO structs |
| `python/sglang/srt/managers/scheduler.py` | `handle_pin_blocks`, `handle_unpin_blocks` scheduler methods; auto-unpin detection when pinned blocks make requests unschedulable |
| `python/sglang/srt/managers/tokenizer_communicator_mixin.py` | `pin_blocks`, `unpin_blocks` communicator methods |

### New Files

| File | Purpose |
|------|---------|
| `test/registered/radix_cache/test_pin_blocks.py` | 11 unit tests (no GPU required) |
| `docs/design/pin_benchmark_v4.py` | E2E benchmark script (single-depth, aiperf-driven) |
| `docs/design/pin_benchmark_v5.py` | Depth-sweep benchmark (parallel GPU, auto-calc flood) |
| `docs/design/results/v5_depth_sweep/` | V5 benchmark results (baseline.json, pinned.json) |
| `docs/design/PIN_POC_WORKLOG.md` | This file |
| `docs/design/PIN_BENCHMARK_REVIEW.md` | Review of earlier (invalid) benchmark attempts |

### External Dependencies

| Path | What |
|------|------|
| `~/datasets/claude_history_sonnet.jsonl` | VIP conversation dataset (1 session, 32 turns, ~29.5K tokens, large system prompt) |
| `~/datasets/long_multiturn_opus.jsonl` | Flood traffic dataset (10 sessions, 19-31 turns each) |
| `~/aiperf/` | NVIDIA aiperf load generator |

---

## How to Test

All commands run from the repo root:

```bash
cd /home/ubuntu/sglang-poc-pin
```

### 1. Unit Tests (no GPU needed)

Tests PIN/UNPIN logic using `RadixCache.create_simulated()`:

```bash
python -m pytest test/registered/radix_cache/test_pin_blocks.py -v -s
```

The `-s` flag shows tree dumps at each step. 11 tests covering: index population, pin/unpin lifecycle, eviction protection, double-pin semantics, ancestor protection.

### 2. Quick Eviction Proof (needs 2x GPU, ~8 min)

Demonstrates that unpinned blocks are naturally evicted under cache pressure. Starts a server with a small cache (18K tokens), sends a request, caches it, floods with competing traffic, then shows the cached blocks were evicted:

```bash
.venv/bin/python -c "
import json, time, requests, subprocess, sys, os, signal
from pathlib import Path

PORT = 30000
BASE = f'http://localhost:{PORT}'
MODEL = 'Qwen/Qwen3-14B'

# Start server with tight cache
cmd = [sys.executable, '-m', 'sglang.launch_server',
       '--model-path', MODEL, '--port', str(PORT),
       '--mem-fraction-static', '0.35', '--tp-size', '2',
       '--trust-remote-code', '--log-level', 'info',
       '--watchdog-timeout', '600']
env = os.environ.copy()
os.makedirs('/tmp/flashinfer_workspace', exist_ok=True)
env['FLASHINFER_WORKSPACE_BASE'] = '/tmp/flashinfer_workspace'
proc = subprocess.Popen(cmd, stdout=open('/tmp/sglang_proof.log','w'),
                        stderr=subprocess.STDOUT, env=env)

# Wait for healthy
for _ in range(150):
    try:
        if requests.get(f'{BASE}/health', timeout=5).status_code == 200: break
    except: pass
    time.sleep(2)
print('Server ready')

# Load VIP system prompt
with open(Path.home() / 'datasets/claude_history_sonnet.jsonl') as f:
    data = json.loads(f.readline())
turns = data['turns']
msgs = [{'role': 'system', 'content': next(t for t in turns if t['role']=='system')['text']},
        {'role': 'user', 'content': next(t for t in turns if t['role']=='user')['text']}]
payload = {'model': 'qwen3-14b', 'messages': msgs, 'max_tokens': 1, 'stream': False}

# Cold request
t0 = time.perf_counter()
requests.post(f'{BASE}/v1/chat/completions', json=payload, timeout=120)
cold = (time.perf_counter() - t0) * 1000
print(f'Cold:        {cold:.0f}ms')

# Warm request (cached)
t0 = time.perf_counter()
requests.post(f'{BASE}/v1/chat/completions', json=payload, timeout=120)
warm = (time.perf_counter() - t0) * 1000
print(f'Cached:      {warm:.0f}ms  ({cold/warm:.1f}x faster)')

# Flood with competing traffic
print('Flooding with 200 requests...')
subprocess.run(['uv','run','aiperf','profile','--model','qwen3-14b','--url',BASE,
    '--endpoint-type','chat','--input-file',str(Path.home()/'datasets/long_multiturn_opus.jsonl'),
    '--custom-dataset-type','multi-turn','--use-server-token-count','--use-legacy-max-tokens',
    '--concurrency','4','--ui-type','none','--no-gpu-telemetry','--no-server-metrics',
    '--output-artifact-dir','/tmp/proof_flood','--profile-export-prefix','flood',
    '--streaming','--osl','64','--request-count','200'],
    capture_output=True, cwd=str(Path.home()/'aiperf'), timeout=1200)

# Post-flood request
t0 = time.perf_counter()
requests.post(f'{BASE}/v1/chat/completions', json=payload, timeout=120)
post = (time.perf_counter() - t0) * 1000
print(f'After flood: {post:.0f}ms  ({'EVICTED' if post > warm*3 else 'CACHED'})')

proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)
subprocess.run(['pkill','-9','-f','sglang.launch_server'], capture_output=True, timeout=5)
"
```

Expected output:
```
Cold:        ~360ms
Cached:      ~100ms  (3.6x faster)
After flood: ~360ms  (EVICTED)
```

### 3. Depth-Sweep Benchmark V5 (needs 2 GPUs, ~90 min)

Runs baseline and pinned phases in **parallel on separate GPUs**, testing PIN at multiple conversation depths. This is the primary benchmark.

```bash
# Baseline on GPU 0
.venv/bin/python docs/design/pin_benchmark_v5.py \
    --phase baseline --gpu 0 --port 30000 --zmq-port 5557 \
    --mem-fraction 0.50 \
    --output-dir /tmp/pin_v5_bl &

# Pinned on GPU 1
.venv/bin/python docs/design/pin_benchmark_v5.py \
    --phase pinned --gpu 1 --port 30001 --zmq-port 5558 \
    --mem-fraction 0.50 \
    --output-dir /tmp/pin_v5_pin &

wait
```

**What it does per depth (D = 0, 2, 6, 10, 16):**

```
1. Start fresh server (one per GPU, mem_fraction=0.50, 42K token cache)
2. [Pinned only] Start KV event collector (ZMQ subscriber)
3. Warmup: send turns 0..D as chat completion (populates radix tree)
4. [Pinned only] Collect block hashes from BlockStored events, POST /pin_blocks
5. Flood: auto-calculated requests via aiperf (cycles cache 3x)
6. Drain server (wait for all in-flight requests to complete)
7. Measure: send turns 0..D+1 as streaming chat completion, capture TTFT
8. Stop server
```

**Auto-calculated flood intensity**: `ceil(cache_capacity * 3 / 150)` requests. The constant 150 is the empirical average unique tokens per flood request in the radix tree (accounts for prefix sharing in multi-turn datasets).

**Key parameters:**
- `--depths`: Conversation depths to test (default: 0 2 6 10 16)
- `--mem-fraction`: GPU memory for KV cache (0.50 gives 42K tokens)
- `--phase`: Run baseline, pinned, or both
- `--gpu`: GPU ID (CUDA_VISIBLE_DEVICES)

**Output files:**
- `<output-dir>/results.json` -- all depth results with TTFT, cached tokens, blocks pinned
- `<output-dir>/params.json` -- benchmark parameters
- `<output-dir>/depth_N/` -- per-depth aiperf artifacts

### 4. V4 Benchmark (legacy, needs 2 GPUs, ~45 min)

Earlier benchmark that measures only system prompt (turn 0) TTFT across 3 reps:

```bash
.venv/bin/python docs/design/pin_benchmark_v4.py \
    --num-reps 3 --mem-fraction 0.40 \
    --flood-concurrency 4 --flood-osl 64 \
    --flood-requests 200 --output-dir /tmp/pin_benchmark_v4
```

---

## What Was Built

### 1. Block Hash Index (`radix_cache.py`)

**Problem**: The router knows blocks by their int64 hash (from KV events). RadixCache has no way to look up a TreeNode by hash.

**Solution**: Added `block_hash_index: Dict[int, TreeNode]` -- a reverse lookup from int64 block hash to TreeNode, populated lazily when `_record_store_event()` computes hashes.

- `_index_node_hashes(node)`: For each hash in `node.hash_value`, adds `hash_str_to_int64(h) -> node` to index
- `_unindex_node_hashes(node)`: Removes entries from index
- Hooked into `_record_store_event()` (after hash computation) and `_record_remove_event()` (before event emission)
- Cleared in `reset()`

### 2. PIN / UNPIN Methods (`radix_cache.py`)

```python
def pin_blocks(self, block_hashes: List[int]) -> int:
    """Pin blocks by hash to resist eviction. Returns count pinned."""
```

- Looks up node via `block_hash_index`
- First pin: calls `inc_lock_ref(node)` which walks up to root incrementing `lock_ref` on each ancestor
- Tracks external pins in `external_pin_count: Dict[int, int]` (separate from request-based lock_ref)
- Subsequent pins increment `external_pin_count` without additional `inc_lock_ref` calls

```python
def unpin_blocks(self, block_hashes: List[int]) -> int:
    """Unpin blocks by hash. Returns count unpinned."""
```

- Decrements `external_pin_count`
- Last unpin (count reaches 0): calls `dec_lock_ref(node)` to restore evictability
- Cleans up `external_pin_count` entry when count reaches 0

### 3. HTTP Endpoints

```
POST /hicache/pin_blocks    {"block_hashes": [int64, ...]}  -> {"pinned_count": N}
POST /hicache/unpin_blocks  {"block_hashes": [int64, ...]}  -> {"unpinned_count": N}
```

Wired through: http_server.py -> io_struct -> tokenizer_communicator -> scheduler -> radix_cache.

### 4. Scheduler Auto-Unpin Safety Net (`scheduler.py`)

When pinned blocks consume too much cache, a request can become mathematically unschedulable: `protected_size > total_tokens - request_tokens`. The scheduler detects this and auto-unpins all blocks to recover:

```python
if (len(self.waiting_queue) > 0
    and self.running_batch.is_empty()
    and self.tree_cache.protected_size() > 0):
    protected = self.tree_cache.protected_size()
    if protected > self.max_total_num_tokens - first_req.extend_input_len:
        self.tree_cache.unpin_blocks(pinned_hashes)
        logger.warning(f"Auto-unpinned {unpinned} blocks: request needs ...")
```

This prevents the scheduler from spinning indefinitely. The detection is math-based (no hardcoded timeouts).

### 5. Unit Tests (`test/registered/radix_cache/test_pin_blocks.py`)

11 tests, all passing. Uses `RadixCache.create_simulated()` -- no GPU required.

| Test | What it proves |
|------|---------------|
| `test_block_hash_index_populated_on_insert` | Index is populated when nodes are inserted |
| `test_block_hash_index_multiple_pages` | Multi-page inserts create multiple index entries |
| `test_index_cleared_on_reset` | `reset()` clears both index and pin counts |
| `test_index_cleared_on_evict` | Evicting a node removes its hashes from index |
| `test_pin_unknown_hash` | Pinning nonexistent hash returns 0, no crash |
| `test_unpin_unknown_hash` | Unpinning nonexistent hash returns 0, no crash |
| `test_pin_sets_lock_ref_and_removes_from_evictable` | PIN sets lock_ref > 0 AND removes from evictable_leaves |
| `test_unpin_restores_evictable_status` | UNPIN restores lock_ref to 0 AND re-adds to evictable_leaves |
| `test_pin_survives_eviction_unpin_does_not` | Insert A+B, pin A, evict(999): A survives, B removed. Unpin A, evict: A removed. |
| `test_double_pin_requires_double_unpin` | Pin x2 -> unpin x1 -> evict fails -> unpin x1 -> evict succeeds |
| `test_pin_protects_ancestors` | Pinning leaf also protects parent via lock_ref propagation |

---

## Key Design Decisions

1. **PIN at RadixCache, not HiRadixCache**: All the required infrastructure (`lock_ref`, `evictable_leaves`, `_update_leaf_status`, hash computation) lives in RadixCache. HiRadixCache inherits everything. `create_simulated()` enables GPU-free testing.

2. **Reuse lock_ref instead of new pinned flag**: `lock_ref` is already checked by `_update_leaf_status()` which gates `evictable_leaves` membership. No new eviction logic needed.

3. **external_pin_count separates router pins from request pins**: Without this, `dec_lock_ref` from a finishing request could accidentally unpin a router-pinned block. The dict tracks how many times each hash was externally pinned.

4. **Lazy index population**: `block_hash_index` is populated in `_record_store_event()` where hashes are already being computed. No extra hash computation needed.

5. **Not a mixin**: PIN is tightly coupled to RadixCache internals (lock_ref, evictable_leaves, _record_store_event). A mixin would add MRO complexity without reducing coupling. Total addition is ~60 lines.

---

## Benchmark Results

### V5 Depth-Sweep (final, Qwen/Qwen3-14B-FP8, mem_fraction=0.50)

**Model**: Qwen/Qwen3-14B-FP8, TP=1, page_size=64
**Cache**: 42,816 tokens (mem_fraction=0.50)
**Flood**: auto-calculated 857 requests/depth (cycles cache 3x), concurrency=8
**Hardware**: 2x GPU (baseline on GPU 0, pinned on GPU 1)

| Depth | Prompt Tokens | BL TTFT | PIN TTFT | Speedup | BL cached | PIN cached | Blocks Pinned |
|-------|--------------|---------|----------|---------|-----------|------------|---------------|
| 0 | 3,158 | 421.5ms | 53.1ms | **7.9x** | 0% | 93% | 46 |
| 2 | 4,810 | 671.9ms | 244.8ms | **2.7x** | 0% | 65% | 74 |
| 6 | 7,347 | 1,055.1ms | 231.6ms | **4.6x** | 0% | 81% | 114 |
| 10 | 10,826 | 1,816.7ms | 227.4ms | **8.0x** | 0% | 89% | 168 |
| 16 | 15,002 | 2,745.1ms | 248.1ms | **11.1x** | 0% | 93% | 233 |

**Key observations:**
- Baseline TTFT grows linearly with depth (more tokens to prefill from scratch)
- Pinned TTFT stays flat at ~230-250ms regardless of depth (only new turn tokens need prefill)
- Depth 0 pinned is especially fast (53ms) because the new turn adds very few tokens on top of the cached system prompt
- Baseline shows 0% cached at all depths (857-request flood fully evicts the 42K cache)
- Speedup scales from 2.7x at depth 2 to 11.1x at depth 16

**PIN cached < 100% explanation**: The Qwen3 chat template wraps the last assistant message with `<think></think>` tags when it's the final message in the list (warmup), but not when followed by more messages (measurement). This causes a prefix match divergence at the last assistant message boundary. This is a Qwen3-specific template issue, not a PIN limitation. On non-thinking models (Llama, Mistral), cache hit would be ~99%.

Raw results: `docs/design/results/v5_depth_sweep/baseline.json` and `pinned.json`.

### V4 Legacy (Qwen/Qwen3-14B-FP8, mem_fraction=0.40)

Earlier benchmark measuring only system prompt (turn 0) TTFT across 3 reps.

**Mean Turn 0 TTFT**: Baseline 354ms, Pinned 42ms. **Speedup: 8.4x**

---

## Gotchas Discovered

1. **`node.evicted` is unreliable after eviction**: `evict()` calls `_delete_leaf()` which removes the node from `parent.children` but does NOT set `node.value = None`. The `evicted` property (`value is None`) returns False on a stale reference. Tests verify eviction by checking tree reachability via `_collect_nodes()` instead.

2. **Simulated value creation**: When `InsertParams(value=None)`, `insert()` creates a fake tensor from token IDs (`torch.tensor(key.token_ids)`). So simulated nodes have real tensor values.

3. **Mock allocator needs real device**: `match_prefix()` calls `torch.empty(..., device=self.device)` for empty results. The mock allocator must have `mock_allocator.device = torch.device("cpu")` -- a plain Mock object causes TypeError.

4. **`flush_cache` clears pins**: `reset()` clears `external_pin_count` and `block_hash_index`. A flush destroys all pins. This is correct behavior but means PIN state doesn't survive a cache reset.

5. **Multi-turn probe self-caching**: In multi-turn replays, turns 1+ self-cache from prior turns in the same session. This dilutes PIN benefit in all-turn averages. Always measure turn 0 TTFT as the primary metric.

6. **Thinking model chat template breaks prefix matching**: Qwen3's chat template wraps the last assistant message with `<think>\n\n</think>\n\n` tags when it's the final message. When the same message appears mid-conversation (not last), no tags are added. This causes the warmup and measurement token sequences to diverge at the last assistant message boundary, losing ~1000-1600 tokens of prefix match. Not controllable via `enable_thinking=False` (template ignores it for message content). Non-thinking models (Llama, Mistral) are unaffected.

7. **Scheduler deadlock with excessive pinning**: When pinned blocks exceed `total_tokens - request_tokens`, the scheduler spins indefinitely in `_get_new_batch_prefill_raw` returning None. No errors are logged. Fixed by adding math-based auto-unpin detection (see "Scheduler Auto-Unpin Safety Net" above).

8. **Flood intensity must be calibrated to cache size**: With `mem_fraction=0.50` (42K cache), 300 flood requests at concurrency=8 were insufficient to cycle the LRU cache. Baseline showed 93% cached instead of 0%. Increasing to 857 requests (`ceil(42816 * 3 / 150)`) achieved full eviction. The constant 150 (unique tokens per flood request in the radix tree) was determined empirically.

9. **`measure_ttft` contamination**: An earlier version sent a second non-streaming request to get `cached_tokens`, which re-cached the prompt tokens and contaminated the measurement. Fixed by using `stream_options: {"include_usage": true}` in the streaming request to get usage stats from the same request.

10. **KV event collector must start BEFORE warmup**: BlockStored events are emitted when blocks are first inserted. If the collector starts after warmup, re-sending the same prompt finds everything cached and emits no events. No block hashes are collected, and pinning fails silently.

---

## What's NOT Done Yet

- **No EVICT (evict_descendants)**: The second cache control operation. Same layer, not yet implemented.
- **No HiRadixCache-specific testing**: Tests use RadixCache.create_simulated(). HiRadixCache has additional tier transition logic (L1/L2/L3) that may need index maintenance hooks.
- **Radix tree cache state inspection**: No out-of-band way to query the radix tree state (pinned blocks, eviction pressure, tree structure). Would help diagnose prefix matching issues and support Dynamo router decision-making.
- **Hash-based block lookup**: Current prefix matching stops at the first token divergence, losing all subsequent pinned blocks. A hash-based lookup (using the existing `block_hash_index`) could find blocks regardless of prefix ordering, but would require changes to KV attention (currently assumes contiguous cached prefixes).
