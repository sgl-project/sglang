# Bug Fixes: Paged LoRA

## Summary

Two categories of bugs were found and fixed in the paged LoRA implementation:
1. **Scheduler side-effects** (Bugs 1–3)
2. **Non-deterministic LRU** (Bug 4)

Additionally, a metric-tracking bug (Bug 5) and a module-name resolution bug
(Bug 6) were fixed.

1. **Scheduler side-effects** (Bugs 1–3): `_can_schedule_lora_req` was calling
   mutating methods (`evict_pages`, `ensure_adapter_ready`) that should only run
   during the forward pass.

2. **Non-deterministic LRU** (Bug 4): `time.monotonic()` for page access
   timestamps causes TP ranks to make different eviction decisions.

---

## Bug 1–3: Scheduler Side-Effects in `_can_schedule_lora_req`

### Root Cause

The scheduler's `_can_schedule_lora_req` is meant to be a **predicate** — it
answers "can this request be scheduled?" without mutating global state.  The
paged-LoRA path violated that contract by calling two side-effecting methods:

1. **`evict_pages()`** — modifies `page_table` entries (sets them to -1),
   updates `free_page_indices`, removes entries from `phys_page_to_uid`, and
   increments `page_generation`.

2. **`ensure_adapter_ready()`** — allocates physical pages, copies LoRA
   weights from CPU to GPU via `copy_()`, and updates `page_table`.

These mutations are only safe inside the forward pass (`fetch_new_loras`),
which runs on **every TP rank** and is off the scheduler's critical path.

### Bug 1 — Performance Bottleneck (Scheduler Blocking)

**Trigger:** Any request whose LoRA adapter is not fully resident (pages
missing due to eviction or first use).

**Mechanism:** `_can_schedule_lora_req` runs on the scheduler thread.
`ensure_adapter_ready` → `load_lora_weight_to_pages` →
`_scatter_adapter_weights` executes synchronous `copy_()` from CPU to GPU for
every layer × module combination.  For a typical 32-layer model with 4 target
modules, this is 128 `copy_()` calls blocking the scheduler.

**Impact:** Increased scheduling latency, reduced throughput under
multi-adapter workloads where page-in is frequent.

### Bug 2 — TP > 1 State Divergence Between Ranks

**Trigger:** Multi-GPU serving (TP > 1) with paged LoRA enabled, when a new
adapter's pages are evicted and reloaded.

**Mechanism:** The scheduler only runs on **rank 0**.  When
`_can_schedule_lora_req` calls `evict_pages` and `ensure_adapter_ready`, only
rank 0's `LoRAPagePool` state is modified:

| State | Rank 0 | Rank 1..N |
|---|---|---|
| `page_table[uid]` | `[phys_new]` | `[phys_old]` or `[-1]` |
| `free_page_indices` | updated | stale |
| `phys_page_to_uid` | updated | stale |
| Weight data in pages | new adapter weights | old/zero data |

During the subsequent forward pass, each rank independently launches the
paged Triton kernels.  The kernels read `page_table` to resolve physical
pages — rank 0 reads the new page, other ranks read stale entries.  The
result is **silent numeric divergence** in LoRA outputs across ranks,
which accumulates into incorrect token generation.

**Impact:** Wrong outputs under TP > 1 (correctness bug).

### Bug 3 — Unnecessary Eviction on Scheduling Failure

**Trigger:** `_can_schedule_lora_req` evicts pages but a subsequent check
fails (e.g., `ensure_adapter_ready` fails because eviction freed enough pages
but the adapter's weights are somehow unavailable, or a later validation in
`validate_lora_batch` fails).

**Mechanism:**
1. `evict_pages(n, protected)` — evicts `n` pages, sets their `page_table`
   entries to -1, removes them from `phys_page_to_uid`.
2. `ensure_adapter_ready(...)` — fails (returns `False`).
3. `_can_schedule_lora_req` returns `False`.

The evicted pages' data is gone, but the scheduling attempt that triggered
the eviction was unsuccessful — the adapter is not scheduled, and the evicted
pages' previous owners now have missing pages that will require costly
re-loading on the next access.

**Impact:** Reduced cache efficiency, increased page-in I/O under contention.

### Fix (Bugs 1–3)

Replace the side-effecting checks in `_can_schedule_lora_req` with a **dry-run
predicate** (`can_ensure_adapter_ready`), and let the actual page allocation,
eviction, and weight loading happen during the forward pass in `fetch_new_loras`
(which already calls `ensure_adapter_ready` and runs on every TP rank).

#### Changes

**1. `python/sglang/srt/lora/paged_mem_pool.py`** — New dry-run methods:

- **`_count_evictable_pages(protected_pages)`**: Counts physical pages that
  are in-use, not in `protected_pages`, and not pinned.  Pure computation.

- **`can_ensure_adapter_ready(uid, rank, protected_pages)`**: Returns `True`
  if the adapter is already complete **or** there are enough free + evictable
  pages to satisfy the missing pages.  Does not allocate, evict, or copy
  weights.

**2. `python/sglang/srt/managers/scheduler.py`** — Dry-run check:

Replace the paged-LoRA block in `_can_schedule_lora_req`:
- **Remove:** `pool.evict_pages(...)` call
- **Remove:** `pool.ensure_adapter_ready(...)` call
- **Replace with:** `pool.can_ensure_adapter_ready(req.lora_id, rank, protected)`

The actual eviction + loading is already performed during the forward pass by
`LoRAManager.fetch_new_loras()` (called from `ForwardBatch.init_new`), which:
1. Runs on **every TP rank** — no state divergence.
2. Is **off the scheduler critical path** — does not block scheduling.
3. Calls `ensure_adapter_ready` for each new adapter, which internally
   handles eviction if needed and loads the weights.

---

## Bug 4 — Non-Deterministic LRU Eviction via `time.monotonic()`

### Root Cause

`LoRAPagePool` used `time.monotonic()` to timestamp page accesses for LRU
eviction ordering.  `time.monotonic()` returns a **wall-clock value** that
differs across processes — each TP rank is an independent process with its
own clock and its own thread-scheduling jitter.

### Trigger

Multi-GPU serving (TP > 1) with paged LoRA enabled, when pages need to be
evicted (i.e. the page pool is under contention with multiple adapters).

### Mechanism

Each rank executes the same logical sequence of page accesses (same
`fetch_new_loras` → `ensure_adapter_ready` → `page_in` → `mark_page_accessed`
calls), but the wall-clock timestamps assigned by `time.monotonic()` differ
across ranks due to:

- Different process start times → different monotonic clock baselines
- Thread-scheduling jitter → two consecutive `time.monotonic()` calls on
  rank 0 may return values 1 µs apart, but on rank 1 they may be 3 µs apart
- OS-level timer interrupt coalescing

The result is that when `evict_pages` sorts candidates by timestamp:

```
candidates.sort(key=lambda p: self.page_access_times[p])
```

different ranks can produce **different sort orders** for pages A and B:

| Rank | `page_access_times[A]` | `page_access_times[B]` | LRU order |
|---|---|---|---|
| Rank 0 | 1000.001 | 1000.002 | A evicted first |
| Rank 1 | 1000.003 | 1000.004 | A evicted first ✓ |
| Rank 2 | 1000.009 | 1000.008 | **B evicted first** ✗ |

Rank 2 evicts a different page than ranks 0 and 1.  From that point forward:

1. `page_table` contents diverge across ranks
2. Paged kernels read different physical pages → **LoRA outputs differ numerically**
3. Accumulated divergence in hidden states can cause all-reduce mismatches →
   **hang or crash**

### Fix (Bug 4)

Replace `time.monotonic()` with a deterministic **logical access counter**
(`_access_counter`) that increments by 1 on every `mark_page_accessed` or
`mark_adapter_pages_accessed` call.

#### Why the logical counter is deterministic

Every TP rank executes the **identical sequence** of LoRA operations during
the forward pass:

1. `fetch_new_loras` → calls `mark_adapter_pages_accessed` for each running
   adapter, then `ensure_adapter_ready` for each new adapter
2. `ensure_adapter_ready` → `page_in` → `mark_page_accessed` for each
   newly allocated page

Since the code path is identical and single-threaded per rank, each rank's
`_access_counter` increments the same number of times in the same order,
assigning identical values to identical pages.  LRU sorting is then bitwise
identical across all ranks.

#### Changes

**`python/sglang/srt/lora/paged_mem_pool.py`**:

| Location | Before | After |
|---|---|---|
| `__init__` | `page_access_times: List[float] = [0.0] * total_pages` | `_access_counter: int = 0` + `page_access_times: List[int] = [0] * total_pages` |
| `mark_page_accessed` | `page_access_times[idx] = time.monotonic()` | `_access_counter += 1; page_access_times[idx] = _access_counter` |
| `mark_adapter_pages_accessed` | `now = time.monotonic()` then assign | `_access_counter += 1; now = _access_counter` then assign |
| Top of file | `import time` | removed |

---

---

## Bug 5 — `total_bytes_evicted` Metric Never Updated

### Root Cause

`total_bytes_evicted` was declared in `__init__` (line 99) and documented as
"Eviction statistics for I7 metric", but `evict_pages()` never wrote to it.
The counter stayed at 0 for the entire lifetime of the pool, making eviction
I/O monitoring useless.

### Trigger

Any eviction.  The metric is incorrect regardless of TP size or workload.

### Mechanism

`evict_pages()` (lines 364–404) successfully evicts pages — frees them,
updates `page_table` entries to -1, removes entries from
`phys_page_to_uid` — but never increments `total_bytes_evicted`.  The
field is a dead write in `__init__` with no reader or writer afterward.

### Fix (Bug 5)

After the eviction loop in `evict_pages()`, add:

```python
if evicted:
    self.total_bytes_evicted += len(evicted) * self._compute_page_bytes()
```

`_compute_page_bytes()` returns the byte count per physical page by summing
the element counts × element sizes across all A/B page buffers.  When
`evicted` is empty (no pages evicted), the counter is untouched.

### Changes

**`python/sglang/srt/lora/paged_mem_pool.py`** — `evict_pages()`:

| Location | Before | After |
|---|---|---|
| After eviction loop (before `page_generation` increment) | *(nothing)* | `if evicted: self.total_bytes_evicted += len(evicted) * self._compute_page_bytes()` |

**`test/registered/unit/lora/test_paged_mem_pool.py`** — new test cases:
- `test_evict_updates_total_bytes`: eviction increments the counter
- `test_evict_zero_pages_no_bytes_update`: zero-length eviction is a no-op

---

---

## Bug 6 — Substring Matching in `update_lora_info` Module Resolution

### Root Cause

`update_lora_info()` in `lora_manager.py` used `pages_key in module_name`
(substring matching) to map page-pool keys to LoRA module names.  This is
ambiguous when module names overlap — a shorter key matches a longer name
that happens to contain it as a substring.

### Trigger

Two or more target modules whose normalized names are substrings of each
other, e.g.:

| Page-pool key | Module name | `pages_key in module_name`? |
|---|---|---|
| `"up_proj"` | `"gate_up_proj"` | **True** (wrong!) |
| `"gate_up_proj"` | `"gate_up_proj_moe"` | **True** (wrong!) |

The `break` on first match makes it dependent on dict iteration order, which
is insertion order in Python 3.7+.  If `"up_proj"` is iterated before
`"gate_up_proj"`, the `gate_up_proj` module gets `A_pages`/`B_pages` from
the wrong memory pool buffer.

### Mechanism

1. `page_pool.A_pages.keys()` is iterated in insertion order.
2. First key that is a substring of `module_name` wins (immediate `break`).
3. The module's `A_pages`/`B_pages` reference is set to the wrong page buffer.
4. At forward time, the paged kernel reads weight data from the wrong buffer
   → shape mismatch or incorrect LoRA output.

### Fix (Bug 6)

Replace substring matching with `get_target_module_name()`, which uses
**longest-match** resolution (same as `_scatter_adapter_weights`):

```python
# Before (fragile substring match):
for pages_key in self.page_pool.A_pages:
    if pages_key in module_name:
        module.A_pages = ...
        break

# After (robust longest-match):
a_page_keys = set(self.page_pool.A_pages.keys())
try:
    pages_key = get_target_module_name(module_name, a_page_keys)
    module.A_pages = self.page_pool.A_pages[pages_key][layer_id]
except ValueError:
    pass  # module has no paged LoRA
```

`get_target_module_name` iterates all candidate keys and returns the
**longest** one that is a substring — so `"gate_up_proj"` wins over
`"up_proj"` for module `"gate_up_proj"`.  When no key matches (module
not in page pool), `ValueError` is raised and silently skipped.

### Changes

**`python/sglang/srt/lora/lora_manager.py`** — `update_lora_info()`:

| Location | Before | After |
|---|---|---|
| A_pages lookup | `for pages_key in A_pages: if pages_key in module_name: ... break` | `get_target_module_name(module_name, a_page_keys)` with `ValueError` catch |
| B_pages lookup | `for pages_key in B_pages: if pages_key in module_name: ... break` | `get_target_module_name(module_name, b_page_keys)` with `ValueError` catch |

---

## Affected Files

| File | Change |
|---|---|
| `python/sglang/srt/lora/paged_mem_pool.py` | Add `_count_evictable_pages()` and `can_ensure_adapter_ready()`; replace `time.monotonic()` LRU with `_access_counter`; update `total_bytes_evicted` in `evict_pages()` |
| `python/sglang/srt/managers/scheduler.py` | Replace evict+ensure with dry-run check in `_can_schedule_lora_req` |
| `python/sglang/srt/lora/lora_manager.py` | Replace substring matching with `get_target_module_name()` in `update_lora_info()` |
| `test/registered/unit/lora/test_paged_mem_pool.py` | Add `TestCanEnsureAdapterReady` (10 cases) + `test_evict_updates_total_bytes` + `test_evict_zero_pages_no_bytes_update` |
