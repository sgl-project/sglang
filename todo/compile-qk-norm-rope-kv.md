# Plan: torch.compile QK-Norm + RoPE + KV Cache Update

## Goal

Compile `q_norm + k_norm + rotary_emb + kv_cache_write` as a single
`torch.compile` unit inside `Qwen3MoeAttention`, using the existing
`--enable-torch-compile --torch-compile-override-layers XXXX --torch-compile-scope local`
flag surface.

Today the fused `sgl_kernel.fused_qk_norm_rope` path fuses norm+rope but
**loses** the fused KV-cache write that the non-fused fallback gets via
`fused_set_kv_buffer_arg` inside `rotary_emb`. A `torch.compile` unit that
covers norm+rope+kv-write can match or beat the hand-written kernel while
being maintainable and extensible.

---

## Status Quo

### `Qwen3MoeAttention.apply_qk_norm_rope` (lines 567-621)

```
qkv_proj ──► split q/k/v
                │
       ┌────────┴────────┐
       │  fused path      │  non-fused fallback
       │  (bf16 only)     │
       │                  │
       │  fused_qk_norm   │  apply_qk_norm(q, k, q_norm, k_norm)
       │  _rope(qkv)      │        │
       │    norm+rope      │  rotary_emb(q, k, fused_set_kv_buffer_arg=...)
       │    NO kv write    │        │  rope + optional kv write
       │                  │        │
       └────────┬────────┘
                │
          forward_core
           save_kv_cache = True if fused path was used
                           (kv write not done yet)
```

### How `--torch-compile-scope local` works today

1. `_to_torch` walks all `nn.Module` children recursively.
2. For each `MultiPlatformOp` subclass, calls `enter_torch_compile`.
3. In `local` mode with `override_layers`, if `class.__name__` is in
   the set, replaces `_forward_method` with
   `torch.compile(self.forward_native, ...)`.
4. Top-level `model.forward` stays **eager** (no outer compile).

**Key limitation:** the unit of compilation is a single `MultiPlatformOp`
subclass (e.g. `RMSNorm`, `RotaryEmbedding`). There is no mechanism to
compile an arbitrary *sub-function* that spans multiple ops inside a
non-`MultiPlatformOp` module like `Qwen3MoeAttention`.

---

## Design: Compilable Attention Sub-regions

### Core Idea

Add a **`CompilableRegion`** mechanism that lets any `nn.Module` register
named sub-functions as individually compilable regions. These regions are
discovered and compiled via the same `_to_torch` walk + override-layer
machinery, alongside existing `MultiPlatformOp` handling.

### Why not just make `Qwen3MoeAttention` a `MultiPlatformOp`?

`MultiPlatformOp` is designed for leaf-level ops with a single `forward`
dispatch. Attention is a composite module with sub-modules (`qkv_proj`,
`q_norm`, `k_norm`, `rotary_emb`, `attn`, `o_proj`). We don't want to
compile the whole attention — just the norm+rope+kv-write slice.

### Interface

```python
# In layers/utils/multi_platform.py or a new layers/utils/compilable_region.py

class CompilableRegionMixin:
    """Mixin for nn.Modules that want to expose named sub-regions
    for torch.compile under --torch-compile-scope local."""

    def get_compilable_regions(self) -> dict[str, str]:
        """Return {region_name: method_name} for compilable sub-functions.

        region_name is what goes in --torch-compile-override-layers.
        method_name is the bound method to compile.
        """
        return {}

    def enter_region_compile(
        self,
        region_name: str,
        compile_options: dict | None = None,
        compile_dynamic: bool = False,
    ):
        """Replace the named method with its torch.compiled version."""
        regions = self.get_compilable_regions()
        method_name = regions[region_name]
        original = getattr(self, method_name)
        compiled = torch.compile(original, options=compile_options,
                                 dynamic=compile_dynamic)
        # Stash original for leave_region_compile
        if not hasattr(self, "_compiled_region_originals"):
            self._compiled_region_originals = {}
        self._compiled_region_originals[region_name] = (method_name, original)
        setattr(self, method_name, compiled)

    def leave_region_compile(self, region_name: str):
        method_name, original = self._compiled_region_originals.pop(region_name)
        setattr(self, method_name, original)
```

### Registering a region in `Qwen3MoeAttention`

```python
class Qwen3MoeAttention(nn.Module, CompilableRegionMixin):

    def get_compilable_regions(self):
        return {
            "QKNormRope": "_qk_norm_rope_kv",
        }

    def _qk_norm_rope_kv(self, q, k, v, positions, forward_batch):
        """Compilable region: qk-norm + rope + kv-cache-write."""
        q, k = apply_qk_norm(
            q=q, k=k,
            q_norm=self.q_norm, k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=None,  # no stream tricks under compile
        )
        q, k = self.rotary_emb(
            positions, q, k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v, layer=self.attn,
                    forward_batch=forward_batch,
                )
                if enable_fused_set_kv_buffer(forward_batch)
                and self.compatible_with_fused_kv_buffer
                else None
            ),
        )
        return q, k

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._qk_norm_rope_kv(q, k, v, positions, forward_batch)
        self._used_fused_qk_norm_rope_last_call = False
        return q, k, v
```

The old `fused_qk_norm_rope` code path is removed (or kept behind a
separate flag for comparison). The compiled region now covers norm + rope +
kv-write — strictly more work fused than the old sgl_kernel path.

### Hooking into `_to_torch`

Extend the recursive walk in `cuda_graph_runner._to_torch` to also handle
`CompilableRegionMixin`:

```python
def _to_torch(model, reverse, num_tokens, compile_scope, override_layers, ...):
    for sub in model._modules.values():
        # Existing MultiPlatformOp handling
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(...)

        # NEW: CompilableRegionMixin handling
        if isinstance(sub, CompilableRegionMixin):
            regions = sub.get_compilable_regions()
            for region_name in regions:
                if override_layers is not None and region_name not in override_layers:
                    continue
                if reverse:
                    sub.leave_region_compile(region_name)
                else:
                    sub.enter_region_compile(
                        region_name,
                        compile_options=compile_options,
                        compile_dynamic=compile_dynamic,
                    )

        # Recurse
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens, ...)
```

### CLI Usage

```bash
python -m sglang.launch_server \
    --enable-torch-compile \
    --torch-compile-scope local \
    --torch-compile-override-layers QKNormRope RMSNorm
```

- `QKNormRope` — compiles the norm+rope+kv region inside each attention layer
- `RMSNorm` — compiles the input/post-attention layernorms (existing path)

Both are resolved by the same `_to_torch` walk. No new CLI flags needed.

---

## Compile Boundary Concerns

### 1. Graph breaks from dynamic control flow

`enable_fused_set_kv_buffer(forward_batch)` queries `forward_batch` state.
If this introduces a graph break:

- **Option A**: Hoist the decision outside the compiled region. Pass a
  pre-computed `fused_set_kv_buffer_arg` (or `None`) into the region.
  The compiled function sees a fixed-shape tensor or None, no dynamic
  branch.

- **Option B**: Use `torch.cond` if the branch is data-dependent and both
  paths are compilable.

Option A is strongly preferred — the decision is batch-level, not
token-level, so it can live in `apply_qk_norm_rope` before calling the
compiled region.

### 2. `apply_qk_norm` internals

`apply_qk_norm` in `models/utils.py` has its own dispatch:
- JIT `fused_inplace_qknorm` (registered as a custom op — compilable)
- Fallback: `q_norm(q.reshape(-1, head_dim))` — calls `RMSNorm.forward`

Under `torch.compile`, the `RMSNorm` inside the region is a sub-module.
Its `_forward_method` will already have been swapped to `forward_native` by
`_to_torch` (if `RMSNorm` is in `override_layers`). If not, the custom op
path (`fused_inplace_qknorm`) is already registered with
`register_custom_op(mutates_args=...)` which Dynamo can handle.

**Recommendation**: inline the norm logic directly in `_qk_norm_rope_kv` to
avoid going through `MultiPlatformOp.forward` dispatch (which has
`_forward_method` indirection that may confuse the compiler):

```python
def _qk_norm_rope_kv(self, q, k, v, positions, fused_kv_arg):
    bs = q.size(0)
    q = q.view(bs, -1, self.head_dim)
    k = k.view(bs, -1, self.head_dim)
    # Direct native RMSNorm — no MultiPlatformOp dispatch
    q = torch.nn.functional.rms_norm(q, (self.head_dim,),
                                      self.q_norm.weight,
                                      self.q_norm.variance_epsilon)
    k = torch.nn.functional.rms_norm(k, (self.head_dim,),
                                      self.k_norm.weight,
                                      self.k_norm.variance_epsilon)
    q = q.view(bs, -1)
    k = k.view(bs, -1)
    q, k = self.rotary_emb.forward_native(positions, q, k,
                                           fused_set_kv_buffer_arg=fused_kv_arg)
    return q, k
```

### 3. `rotary_emb` under compile

`RotaryEmbedding` is a `MultiPlatformOp`. Its `forward_native` uses pure
PyTorch ops. Under `local` compile with `QKNormRope` in the override list,
`rotary_emb` itself does NOT need to be separately compiled — it lives
*inside* the compiled region. Call `self.rotary_emb.forward_native()`
directly to bypass the `_forward_method` indirection.

### 4. KV cache write under compile

The fused KV buffer write (`fused_set_kv_buffer_arg`) calls into a JIT
kernel registered via `register_custom_op`. This should be transparent to
Dynamo. If it causes graph breaks, the fallback is to set
`fused_set_kv_buffer_arg=None` and let `forward_core` handle the KV write
via `save_kv_cache=True`. This loses the fusion but keeps correctness.

---

## Implementation Steps

### Phase 1: Infrastructure

1. **Add `CompilableRegionMixin`** to `layers/utils/` (or directly in
   `multi_platform.py`).
2. **Extend `_to_torch`** in `cuda_graph_runner.py` to discover and
   compile regions from `CompilableRegionMixin`.
3. **Unit test**: a mock module with a compilable region, verify it gets
   compiled/uncompiled by `_to_torch` round-trip.

### Phase 2: Qwen3 MoE Attention

4. **Add `CompilableRegionMixin`** to `Qwen3MoeAttention`.
5. **Implement `_qk_norm_rope_kv`** with direct `rms_norm` +
   `rotary_emb.forward_native` calls.
6. **Refactor `apply_qk_norm_rope`**: remove the `fused_qk_norm_rope`
   branch; always use the new region (compiled or not).
7. **Adjust `forward_core`**: `_used_fused_qk_norm_rope_last_call` becomes
   irrelevant since we always go through rope+kv-write path.

### Phase 3: Validation

8. **Correctness**: run `bench_one_batch.py` with
   `--enable-torch-compile --torch-compile-scope local --torch-compile-override-layers QKNormRope`
   and compare logits against baseline.
9. **Performance**: profile decode latency at bs=1 and bs=32, compare
   against `fused_qk_norm_rope` and the non-fused fallback.
10. **Graph breaks**: run with `TORCH_LOGS="graph_breaks"` to verify the
    region compiles cleanly.

### Phase 4: Generalize

11. Apply the same pattern to other models with QK norm (e.g. future
    Qwen variants, Gemma2).
12. Consider a **`@compilable_region("RegionName")`** decorator for
    terser registration.

---

## Open Questions

1. **Can `fused_set_kv_buffer_arg` ops be compiled without graph breaks?**
   Needs empirical test. Fallback is straightforward.

2. **Should the region name be per-class or per-instance?**
   Per-class is simpler and matches `MultiPlatformOp`'s class-name
   matching. Instance-level (e.g. per-layer) can be added later if
   different layers need different compile strategies.

3. **Interaction with `--torch-compile-scope full`.**
   In `full` mode, the outer `model.forward` is compiled. Regions inside
   it would just be normal function calls traced by Dynamo — no double
   compile. The `enter_region_compile` should be a no-op when
   `compile_scope == "full"` to avoid nested `torch.compile`.

4. **Alt-stream overlap for q/k norm.**
   The non-fused path overlaps q_norm and k_norm on separate CUDA streams
   during graph capture. `torch.compile` does its own stream/op
   scheduling. Drop the alt-stream logic inside the compiled region.
