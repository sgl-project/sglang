# Plan: torch.compile RoPE + KV Cache Store

## Goal

Replace the two hand-written CUDA kernels visible in gpt_oss nsys traces:

```
fused_rope_kernel<true, 64, true, __nv_bfloat16, long, 8>(FusedRopeParams)
store_kvcache<1024, 2, true, long>(StoreKVCacheParams)
```

with a **single `torch.compile` unit** that covers RoPE + KV cache scatter,
using the existing flag surface:

```bash
--enable-torch-compile \
--torch-compile-override-layers RotaryEmbedding \
--torch-compile-scope local
```

This demonstrates that `torch.compile` can replace custom JIT kernels for
both **standard** (single-pool) and **SWA** (dual-pool) KV cache writes.
All changes are confined to the `forward_native` compile path — the
eager/JIT kernel path (`forward_cuda`) is left untouched.

---

## Status Quo

### Standard KV cache (e.g. Llama, Qwen)

```
forward_prepare:
  q, k = rotary_emb(positions, q, k,
                     fused_set_kv_buffer_arg=FusedSetKVBufferArg(...))
                     │
                     ▼ forward_cuda → apply_rope_with_cos_sin_cache_inplace
                       ┌─────────────────────────────────────┐
                       │ FusedRopeKernel::run_fused           │  ← 1 kernel
                       │   RoPE on q,k  +  scatter k,v→cache │
                       └─────────────────────────────────────┘

forward_core:
  attn(q, k, v, save_kv_cache=False)   ← KV already written
```

One fused kernel. Works because `enable_fused_set_kv_buffer` returns True
(bf16, non-SWA).

### SWA KV cache (gpt_oss)

`GptOssForCausalLM` is a hybrid SWA model → `SWAKVPool` → fused path
disabled:

```
forward_prepare:
  fused_set_kv_buffer_arg = None       ← enable_fused_set_kv_buffer = False
  q, k = rotary_emb(positions, q, k)
                     │
                     ▼ forward_cuda → apply_rope_inplace
                       ┌──────────────────────┐
                       │ FusedRopeKernel::run  │  ← kernel 1: RoPE only
                       └──────────────────────┘

forward_core:
  attn(q, k, v, save_kv_cache=True)
       │
       ▼ attn_backend → SWAKVPool.set_kv_buffer
         ┌─────────────────────────────┐
         │ store_kvcache                │  ← kernel 2: KV scatter
         │   routes via layers_mapping  │
         │   picks full_kv_pool or      │
         │   swa_kv_pool + swa_loc      │
         └─────────────────────────────┘
```

Two kernels. The fused path is disabled because `create_fused_set_kv_buffer_arg`
uses `out_cache_loc` (full-pool indices), but SWA layers need `out_cache_loc_swa`
(SWA-pool indices).

### `forward_native` today

Both `RotaryEmbedding.forward_native` (runtime) and the sgl_kernel testing
copy assert-fail when `fused_set_kv_buffer_arg is not None`:

```python
# srt/layers/rotary_embedding/base.py:199-201
assert (
    fused_set_kv_buffer_arg is None
), "fused_set_kv_buffer_arg is not supported for native implementation"
```

```python
# sgl-kernel/python/sgl_kernel/testing/rotary_embedding.py:113-115
assert (
    fused_set_kv_buffer_arg is None
), "fused_set_kv_buffer_arg is not supported for native implementation"
```

---

## Design

### Core Idea

1. Implement KV cache scatter in `forward_native` using pure PyTorch ops.
2. Add a compile-only gate that enables fused KV args for SWA models
   (by picking the correct `cache_loc` per layer). The eager/JIT path
   is **not changed** — SWA models continue to use the two-kernel path
   when torch.compile is off.
3. Override `dynamic=True` for the `RotaryEmbedding` compiled function.
4. With `--torch-compile-scope local --torch-compile-override-layers
   RotaryEmbedding`, the existing `_to_torch` machinery compiles
   `forward_native` → Inductor fuses RoPE + scatter into one kernel.

No new CLI flags, no new `CompilableRegion` infra needed.

### Why this works out of the box

The `local` compile scope already does the right thing:

1. `_to_torch` walks all `nn.Module` children.
2. Finds every `RotaryEmbedding` instance (a `MultiPlatformOp` subclass).
3. Since `"RotaryEmbedding"` is in `override_layers`, calls
   `enter_torch_compile` → swaps `_forward_method` to
   `torch.compile(self.forward_native, ...)`.
4. The model's `self.rotary_emb(positions, q, k, fused_set_kv_buffer_arg=...)`
   dispatches through `MultiPlatformOp.forward` → `_forward_method` →
   the compiled `forward_native`.

The only code changes are inside `forward_native`, the
`RotaryEmbedding` compile config, and the model-utils helpers.

---

## Implementation

### Step 1: Implement KV scatter in `forward_native`

In `srt/layers/rotary_embedding/base.py`, replace the assert with actual
scatter logic:

```python
def forward_native(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    query_rot = self._apply_rotary_emb_wrapped(
        query_rot, cos, sin, self.is_neox_style
    )
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)
    key_rot = key[..., : self.rotary_dim]
    key_pass = key[..., self.rotary_dim :]
    key_rot = self._apply_rotary_emb_wrapped(
        key_rot, cos, sin, self.is_neox_style
    )
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    # Fused KV cache scatter — pure PyTorch, compilable by Inductor
    if fused_set_kv_buffer_arg is not None:
        cache_loc = fused_set_kv_buffer_arg.cache_loc
        k_buffer = fused_set_kv_buffer_arg.k_buffer
        v_buffer = fused_set_kv_buffer_arg.v_buffer
        value = fused_set_kv_buffer_arg.value
        k_buffer[cache_loc] = key.view(-1, k_buffer.shape[-1])
        v_buffer[cache_loc] = value.view(-1, v_buffer.shape[-1])

    return query, key
```

The `k_buffer[cache_loc] = ...` is a standard index-put that Inductor
handles natively. Under `torch.compile`, this fuses with the preceding
RoPE arithmetic into a single Triton kernel (or a small number of
pointwise + scatter kernels that the compiler can schedule optimally).

Mirror the same change in the sgl_kernel testing copy at
`sgl-kernel/python/sgl_kernel/testing/rotary_embedding.py`.

### Step 2: Force `dynamic=True` for `RotaryEmbedding` compile

The centralized `compile_dynamic` in `cuda_graph_runner.py` is currently:

```python
compile_dynamic = _is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE")
```

This is `False` on CUDA. But `forward_native` with KV scatter has dynamic
tensor dimensions at every decode iteration: `num_tokens` varies across
CUDA graph buckets, and `positions`/`cache_loc` values change every call.
The compiled function must use `dynamic=True`.

Override `_get_local_torch_compile_forward_method` in `RotaryEmbedding`:

```python
# srt/layers/rotary_embedding/base.py

def _get_local_torch_compile_forward_method(
    self,
    method_name: str,
    compile_options: Optional[dict] = None,
    compile_dynamic: bool = False,
) -> Callable:
    # num_tokens and cache_loc shapes vary across CUDA graph buckets
    return torch.compile(
        getattr(self, method_name),
        options=compile_options,
        dynamic=True,
    )
```

This is self-contained: only `RotaryEmbedding` gets `dynamic=True`.
Other `MultiPlatformOp` subclasses (e.g. `RMSNorm`, `FusedMoE`) continue
to use whatever the centralized `compile_dynamic` flag says.

### Step 3: Enable fused KV arg for SWA models under compile

Keep the original `enable_fused_set_kv_buffer` function and add the
compile query inside it. The eager path is untouched — when
`is_compiled=False` (the default), the `SWAKVPool` exclusion stays.

In `models/utils.py`, extend the existing function:

```python
def enable_fused_set_kv_buffer(
    forward_batch: ForwardBatch,
    is_compiled: bool = False,
):
    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache.
    When is_compiled=True (torch.compile path), also allows SWAKVPool
    since forward_native handles the dual-pool addressing in pure PyTorch."""
    return (
        _is_cuda
        and hasattr(forward_batch.token_to_kv_pool, "dtype")
        and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
        and (is_compiled or not isinstance(forward_batch.token_to_kv_pool, SWAKVPool))
    )
```

And extend `create_fused_set_kv_buffer_arg` to pick the correct
`cache_loc` for SWA layers:

```python
def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
):
    from sglang.jit_kernel.rope import FusedSetKVBufferArg

    layer_id = layer.layer_id
    token_to_kv_pool = forward_batch.token_to_kv_pool

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)
    assert layer.k_scale is None and layer.v_scale is None

    # Pick correct cache_loc for SWA vs full layers
    cache_loc = forward_batch.out_cache_loc
    if isinstance(token_to_kv_pool, SWAKVPool):
        _, is_swa_layer = token_to_kv_pool.layers_mapping[layer_id]
        if is_swa_layer:
            cache_loc = forward_batch.out_cache_loc_swa

    return FusedSetKVBufferArg(
        value=value,
        k_buffer=k_buffer.view(k_buffer.shape[0], -1),
        v_buffer=v_buffer.view(v_buffer.shape[0], -1),
        cache_loc=cache_loc,
    )
```

`SWAKVPool.get_key_buffer(layer_id)` already returns the correct
sub-pool buffer (SWA or full) based on `layers_mapping`. The only
missing piece today is `cache_loc`, which this fixes.

Call sites just pass `is_compiled=self.rotary_emb.is_torch_compile`:

```python
# gpt_oss.py forward_prepare — only change is the is_compiled kwarg
_is_compiled = self.rotary_emb.is_torch_compile
extra_args = {}
if not _is_npu:
    extra_args = {
        "fused_set_kv_buffer_arg": (
            create_fused_set_kv_buffer_arg(
                value=v, layer=self.attn, forward_batch=forward_batch,
            )
            if enable_fused_set_kv_buffer(forward_batch, is_compiled=_is_compiled)
            else None
        ),
    }
```

And in `forward_core`:

```python
save_kv_cache=not enable_fused_set_kv_buffer(forward_batch, is_compiled=_is_compiled),
```

The `is_torch_compile` attribute already exists on every `MultiPlatformOp`
(set by `enter_torch_compile`), so no new flags or functions are needed.

When `is_compiled=False` (default), the function behaves identically to
today — existing callers that don't pass the kwarg are unaffected.

### Step 4: No changes to eager/JIT path

The following are **explicitly not changed**:

- `enable_fused_set_kv_buffer` — keeps the `SWAKVPool` exclusion.
  Eager SWA models continue to use the two-kernel path.
- `forward_cuda` — continues to dispatch to JIT kernels.
- `FusedRopeKernel` / `store_kvcache` kernels — untouched.
- Attention backends — continue to handle `save_kv_cache=True` for
  eager SWA models.

---

## Compile Boundary Concerns

### 1. Dynamic shapes — `dynamic=True` required

`forward_native` receives `positions`, `query`, `key`, `cache_loc` with
a dynamic `num_tokens` dimension that varies across CUDA graph buckets.
The centralized `compile_dynamic` flag in `patch_model` is `False` on
CUDA (gated behind `_is_hip and SGLANG_TORCH_DYNAMIC_SHAPE`):

```python
# cuda_graph_runner.py:468
compile_dynamic = _is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE")
```

We **cannot** rely on this global flag. Instead, `RotaryEmbedding`
overrides `_get_local_torch_compile_forward_method` to force
`dynamic=True` for its own compiled function (Step 2 above). This is
self-contained — other ops keep their existing `compile_dynamic`
behavior.

### 2. FusedSetKVBufferArg as a dataclass

The `fused_set_kv_buffer_arg` is a Python dataclass holding tensor
fields. Dynamo traces through dataclass attribute access without graph
breaks. The `.cache_loc`, `.k_buffer`, `.v_buffer`, `.value` fields are
all tensors — standard Dynamo territory.

When `fused_set_kv_buffer_arg is None`, the `if` branch is a
Python-level guard that Dynamo specializes on. This means two compiled
variants: one with KV scatter (arg is not None), one without. Both are
cached by Dynamo — no runtime overhead from the branch.

### 3. cos_sin_cache as a registered buffer

`self.cos_sin_cache` is a persistent buffer registered on the module.
`index_select` on it is compilable. The buffer is constant across calls
(same positions range), so Inductor can optimize the access pattern.

### 4. What Inductor produces

For the compiled `forward_native` with KV scatter, Inductor sees:

```
index_select(cos_sin_cache, 0, positions)
  → chunk → multiply/add (RoPE math) → cat
  → index_put_(k_buffer, [cache_loc], key)
  → index_put_(v_buffer, [cache_loc], value)
```

**Confirmed result**: Inductor fuses all operations into a **single
Triton kernel** using `SequentialComboKernelGrid` with 3 sub-kernels
(RoPE on query, RoPE on key + k scatter, v scatter). This replaces
both the `fused_rope_kernel` and `store_kvcache` JIT kernels with one
Inductor-generated kernel. See "Verified fusion" in the Implementation
Update section for details.

---

## Files to Change

| File | Change |
|------|--------|
| `srt/layers/rotary_embedding/base.py` | Replace assert with KV scatter in `forward_native`; override `_get_local_torch_compile_forward_method` to force `dynamic=True` |
| `sgl-kernel/python/sgl_kernel/testing/rotary_embedding.py` | Same: replace assert with KV scatter in `forward_native` |
| `srt/models/utils.py` | Add `is_compiled` kwarg to `enable_fused_set_kv_buffer`; extend `create_fused_set_kv_buffer_arg` to pick `out_cache_loc_swa` for SWA layers |
| `srt/models/gpt_oss.py` | Pass `is_compiled=self.rotary_emb.is_torch_compile` to `enable_fused_set_kv_buffer` |

No changes needed in:
- `server_args.py` (existing flags suffice)
- Attention backends (still handle `save_kv_cache=True` for eager SWA)
- JIT kernels / `forward_cuda` (untouched)

**Note:** `cuda_graph_runner.py` was originally expected to require no
changes, but it turned out that `DecodeInputBuffers` needed
`out_cache_loc_swa` support for SWA models — see Fix 4 in the
Implementation Update below.

---

## Validation

### Correctness

```bash
# Standard model (e.g. Qwen3 MoE)
python -m sglang.bench_one_batch \
    --model <qwen3-moe> \
    --enable-torch-compile \
    --torch-compile-scope local \
    --torch-compile-override-layers RotaryEmbedding

# SWA model (gpt_oss)
python -m sglang.bench_one_batch \
    --model <gpt-oss> \
    --enable-torch-compile \
    --torch-compile-scope local \
    --torch-compile-override-layers RotaryEmbedding
```

Compare output logits against baseline (no compile). They should match
within bf16 tolerance.

### Graph breaks

```bash
TORCH_LOGS="graph_breaks" python -m sglang.bench_one_batch \
    --model <gpt-oss> \
    --enable-torch-compile \
    --torch-compile-scope local \
    --torch-compile-override-layers RotaryEmbedding
```

Verify zero graph breaks inside `forward_native`.

### Performance (nsys)

Profile decode at various batch sizes. Expected outcome:

| Path | Kernels | Notes |
|------|---------|-------|
| Eager (current gpt_oss) | `fused_rope_kernel` + `store_kvcache` | 2 kernels, unchanged |
| Eager (current standard) | `fused_rope_kernel::run_fused` | 1 JIT kernel, unchanged |
| Compiled (both) | 1-4 Triton kernels | Inductor-generated |

The compiled path should show comparable or better latency vs the
2-kernel SWA path. The main metric is end-to-end decode latency, not
individual kernel time (Inductor scheduling may differ from hand-tuned
launch configs).

---

## Open Questions

1. ~~**Does Inductor fuse `index_put_` with preceding pointwise?**~~
   **Resolved: Yes.** Inductor fuses RoPE pointwise + `index_put_`
   scatter into a single `SequentialComboKernelGrid` Triton kernel.
   The key scatter for k is fused with the RoPE rotation of k in the
   same sub-kernel. This matches or exceeds the hand-written fused
   kernel for standard models, and replaces 2 kernels → 1 for SWA.

2. **Extend to QK-norm + RoPE + KV?**
   See `todo/compile-qk-norm-rope-kv.md` for the broader plan. This doc
   focuses on the minimal RoPE + KV fusion. The two plans compose: once
   `forward_native` handles KV scatter, the `CompilableRegion` approach
   can wrap norm + rope + kv-write and compile them together.

---

## Implementation Update (2026-03-24)

The plan above has been implemented. This section documents the actual
changes made and issues discovered during implementation.

### What was implemented

All four steps from the plan were completed as designed:

1. **KV scatter in `forward_native`** — `base.py` and the sgl-kernel
   testing copy now perform the scatter when `fused_set_kv_buffer_arg`
   is not None.

2. **`dynamic=True` override** — `RotaryEmbedding` overrides
   `_get_local_torch_compile_forward_method` to force `dynamic=True`.

3. **SWA compile gate** — `enable_fused_set_kv_buffer` accepts
   `is_compiled` kwarg; `create_fused_set_kv_buffer_arg` picks
   `out_cache_loc_swa` for SWA layers.

4. **gpt_oss.py call sites** — `forward_prepare` and `forward_core`
   pass `is_compiled=self.rotary_emb.is_torch_compile`.

### Additional changes beyond the original plan

#### Fix 1: MRO-based override layer matching (`multi_platform.py`)

`--torch-compile-override-layers RotaryEmbedding` did not match
subclasses like `YaRNScalingRotaryEmbedding` because the check used
`self.__class__.__name__` (exact match). Added
`_matches_override_layers` that walks `type(self).__mro__`, so
specifying a base class name now matches all subclasses. Tests added
in `test_multi_platform.py`.

#### Fix 2: `index_put_` instead of `__setitem__` (`base.py`)

**Critical performance bug.** The original plan used
`k_buffer[cache_loc] = value`. During CUDA graph capture, `cache_loc`
was `None` on the first Dynamo trace (because `out_cache_loc_swa` is
not populated by the regular `CudaGraphRunner` — see Fix 3). Python's
`k_buffer[None] = value` silently unsqueezes and broadcasts to the
*entire* buffer. Dynamo encoded this as graph 0 with an
`expand → copy_` pattern that iterated over `pool_size × head_dim`
elements (1.19 billion for gpt-oss-20b), overwriting every row of the
KV pool instead of writing 1 row. This made the compiled path ~100x
slower than the JIT kernel.

Switched to explicit `index_put_`:

```python
k_buffer.index_put_([cache_loc], key.view(-1, k_buffer.shape[-1]))
v_buffer.index_put_([cache_loc], value.view(-1, v_buffer.shape[-1]))
```

This forces the `index_put` decomposition regardless of how Dynamo
traces `cache_loc`, and fails loudly on `None` instead of silently
corrupting the cache.

#### Fix 3: Guard against missing `out_cache_loc_swa` (`utils.py`)

The regular `CudaGraphRunner` did not populate
`forward_batch.out_cache_loc_swa` (only `PiecewiseCudaGraphRunner`
did). For SWA models during CUDA graph capture, `cache_loc` resolved
to `None`. Two layers of defense were added:

1. `enable_fused_set_kv_buffer` now requires
   `forward_batch.out_cache_loc_swa is not None` when operating on an
   `SWAKVPool` under compile. When the SWA loc is missing, the fused
   path is disabled entirely, so both the RoPE scatter and
   `save_kv_cache` flag stay consistent.

2. `create_fused_set_kv_buffer_arg` returns `None` if `cache_loc`
   resolves to `None` (belt-and-suspenders).

#### Fix 4: `out_cache_loc_swa` in `CudaGraphRunner` (`cuda_graph_runner.py`)

With Fix 3, SWA models using the regular `CudaGraphRunner` fell back
to the two-kernel eager path because `out_cache_loc_swa` was `None`
during CUDA graph capture. To fully enable the fused path, the regular
`CudaGraphRunner` was extended to handle `out_cache_loc_swa`,
mirroring the pattern already present in `PiecewiseCudaGraphRunner`:

1. **Allocation**: Added `out_cache_loc_swa: Optional[torch.Tensor]`
   to `DecodeInputBuffers`. `create()` conditionally allocates
   `torch.zeros((max_num_token,), dtype=torch.int64)` when
   `is_hybrid_swa=True`.

2. **Capture**: `capture_one_batch_size()` slices
   `buffers.out_cache_loc_swa[:num_tokens]` and passes it to the
   `ForwardBatch` constructor.

3. **Replay**: `populate_from_forward_batch()` copies the live
   `forward_batch.out_cache_loc_swa` into the buffer, and zeros the
   buffer when `bs != raw_bs` (padding).

This is a small int64 index tensor (`max_num_token` elements × 8
bytes) — negligible memory overhead. The actual KV cache buffers are
not duplicated; both pools (`full_kv_pool`, `swa_kv_pool`) only store
layers assigned to them, and the SWA pool independently recycles old
tokens via `free_swa()` as sequences grow past the sliding window.

### Verified fusion: Inductor output

With all fixes applied, `torch_compile_debug` artifacts confirm that
Inductor produces a **single fused Triton kernel** for both full and
SWA layers. The kernel uses `SequentialComboKernelGrid` with 3
sub-kernels:

| Sub-kernel | Elements | Operation |
|---|---|---|
| `pid < num_xblocks_0` | 4096 (64 heads × 64 dim) | RoPE on **query** |
| `pid < num_xblocks_1` | 512 (8 heads × 64 dim) | RoPE on **key** + `index_put_` into **k_buffer** |
| `pid < num_xblocks_2` | 512 | `index_put_` **value** into **v_buffer** |

Dynamo compiles two graph variants:

- **Graph 0** (full attention layers): k/v buffers are static
  `bf16[2327680, 512]`, `cache_loc` = `out_cache_loc`.
- **Graph 1** (SWA layers): k/v buffers are dynamic
  `bf16[s77, 512]` / `bf16[s67, 512]`, `cache_loc` = `out_cache_loc_swa`.
  Bounds checks use runtime `ks0`/`ks1` kernel arguments.

The two graphs exist because the SWA pool has a different token
capacity than the full pool (controlled by `swa_full_tokens_ratio`,
default 0.8). The recompile from graph 0 → graph 1 happens once at
startup; both are then cached and replayed via CUDA graphs.

### Files changed (final)

| File | Change |
|------|--------|
| `srt/layers/rotary_embedding/base.py` | Replace assert with `index_put_` scatter; override `_get_local_torch_compile_forward_method` for `dynamic=True` |
| `sgl-kernel/python/sgl_kernel/testing/rotary_embedding.py` | Same: replace assert with `index_put_` scatter |
| `srt/models/utils.py` | `is_compiled` kwarg + `out_cache_loc_swa` guard in `enable_fused_set_kv_buffer`; `cache_loc is None` guard in `create_fused_set_kv_buffer_arg` |
| `srt/models/gpt_oss.py` | Pass `is_compiled` kwarg |
| `srt/layers/utils/multi_platform.py` | MRO-based `_matches_override_layers` for subclass matching |
| `srt/model_executor/cuda_graph_runner.py` | `out_cache_loc_swa` in `DecodeInputBuffers`: allocate, capture, and replay for SWA models |
| `test/registered/unit/layers/test_multi_platform.py` | Tests for MRO matching |
| `test/registered/unit/layers/test_fused_kv_buffer.py` | Unit tests for enable/create helpers + numerics vs CUDA kernel |

### Data flow: how `cache_loc` reaches the compiled kernel

#### Non-fused (eager) path — why `out_cache_loc_swa` was never needed

In the eager path, the KV cache write is handled by the attention
backend, which calls `SWAKVPool.set_kv_buffer(layer, loc, k, v)`.
This method receives `loc = out_cache_loc` (full-pool indices) and
internally translates to SWA indices for SWA layers:

```
SWAKVPool.set_kv_buffer(layer, loc, cache_k, cache_v)
  └─ is_swa_layer?
       ├─ yes → loc = self.swa_loc or translate_loc_from_full_to_swa(loc)
       │        swa_kv_pool.set_kv_buffer(loc, cache_k, cache_v)
       └─ no  → full_kv_pool.set_kv_buffer(loc, cache_k, cache_v)
```

The translation is encapsulated inside the pool. The caller (attention
backend) and the `ForwardBatch`/`DecodeInputBuffers` only ever needed
`out_cache_loc`. That's why `CudaGraphRunner` never had
`out_cache_loc_swa`.

#### Fused (compiled) path — why `out_cache_loc_swa` is now required

The compiled `forward_native` writes directly to the pool's buffer:

```python
k_buffer.index_put_([cache_loc], key.view(-1, k_buffer.shape[-1]))
v_buffer.index_put_([cache_loc], value.view(-1, v_buffer.shape[-1]))
```

`k_buffer` is already the correct sub-pool buffer (returned by
`SWAKVPool.get_key_buffer(layer_id)` which routes via
`layers_mapping`). But `cache_loc` must be in the matching coordinate
space — full-pool indices for full layers, SWA-pool indices for SWA
layers. There is no `SWAKVPool.set_kv_buffer()` call inside the
compiled graph to do the translation.

#### Decode step data flow (CUDA graph replay)

```
┌─ Scheduler ─────────────────────────────────────────────────────┐
│ alloc_for_decode()                                              │
│   └─ allocator.alloc_decode() → out_cache_loc (full-pool rows)  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ ForwardBatch.init_new() ──────────────────────────────────────┐
│ out_cache_loc = batch.out_cache_loc                            │
│ out_cache_loc_swa = translate_loc_from_full_to_swa(            │
│                         out_cache_loc)                          │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ CudaGraphRunner.replay() ─────────────────────────────────────┐
│ populate_from_forward_batch():                                  │
│   buffers.out_cache_loc[:N].copy_(forward_batch.out_cache_loc)  │
│   buffers.out_cache_loc_swa[:N].copy_(                          │
│       forward_batch.out_cache_loc_swa)                          │
│                                                                 │
│ torch.cuda.CUDAGraph.replay()                                   │
│   └─ per layer: compiled forward_native reads from the same     │
│      GPU addresses that buffers.out_cache_loc[_swa] point to    │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Compiled forward_native (Triton kernel) ──────────────────────┐
│ For full attention layer:                                       │
│   cache_loc = out_cache_loc        (from FusedSetKVBufferArg)   │
│   k_buffer  = full_kv_pool.k_buffer[layer_idx]                  │
│   k_buffer.index_put_([cache_loc], rotated_key)                 │
│                                                                 │
│ For SWA layer:                                                  │
│   cache_loc = out_cache_loc_swa    (from FusedSetKVBufferArg)   │
│   k_buffer  = swa_kv_pool.k_buffer[swa_layer_idx]              │
│   k_buffer.index_put_([cache_loc], rotated_key)                 │
└─────────────────────────────────────────────────────────────────┘
```

The key property that makes CUDA graph replay work: the
`DecodeInputBuffers` tensors are allocated once during `__init__` and
their GPU memory addresses are baked into the captured graph. Each
replay, `populate_from_forward_batch()` overwrites the *contents*
(via `.copy_()`) before `graph.replay()`, so the Triton kernel sees
fresh indices at the same addresses without re-recording.

### SWA memory layout

`SWAKVPool` creates two separate `MHATokenToKVPool` instances:

```
full_kv_pool:  [full_max_total_num_tokens, head_num, head_dim] × full_layers_num
swa_kv_pool:   [swa_max_total_num_tokens,  head_num, head_dim] × swa_layers_num
```

Each layer's KV data lives in exactly one pool — no duplication. The
`swa_full_tokens_ratio` (default 0.8) controls the token capacity
split. The SWA pool can be smaller because old tokens beyond the
sliding window are independently recycled via `free_swa()` (called by
the scheduler and the radix cache eviction), while full-pool tokens
persist for the lifetime of the sequence.

### Remaining validation

End-to-end validation with `bench_one_batch` for both standard and SWA
models (as described in the Validation section above) should be run to
confirm the compiled path produces correct output and no regressions.
