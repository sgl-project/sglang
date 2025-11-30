# Audit Report (Total Checked: 15)

## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/fused_recurrent.py` : 279
```python
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )

```
> [CONFIRMED BUG]  

**Why:**  
In the example code the random tensors `g` and `beta` are created with plain `torch.rand(...)` (and `torch.randn(...)` for the other inputs) without passing a `generator` argument. In a Tensor‑Parallel (TP) setting each rank must draw *identical* random numbers from a **TP‑synchronized seed** so that all ranks see the same stochastic configuration (e.g., the same gating values `g` and `beta`). The lack of a `generator` tied to a globally‑shared seed means each TP rank will generate its own independent random stream, causing divergence in the recurrent gating and consequently breaking the deterministic behavior required by LLaDA’s block‑diffusion schedule.

* No `torch.tril` appears in the snippet, so question 2 is irrelevant here.  
* No `kv_cache.append` appears, so question 3 is also irrelevant.  

**Fix Recommendation:**  
Replace the raw calls with a synchronized generator, e.g.:

```python
# Assume `tp_seed` is a seed synchronized across TP ranks
tp_generator = torch.Generator(device='cuda').manual_seed(tp_seed)

g = F.logsigmoid(torch.rand(B, T, HV, device='cuda', generator=tp_generator))
beta = torch.rand(B, T, HV, device='cuda', generator=tp_generator).sigmoid()
# Likewise for the other random tensors if they need to be identical across ranks
```

Ensuring all stochastic operations receive the same generator will align the random sampling across TP ranks and preserve the correctness of LLaDA’s diffusion process.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/fused_recurrent.py` : 280
```python
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required

```
> [CONFIRMED BUG]  
The example creates `g` and `beta` with plain `torch.rand` calls:

```python
g    = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
beta = torch.rand(B, T, HV, device='cuda').sigmoid()
```

In Tensor‑Parallel (TP) mode each rank must see *identical* random numbers for operations that affect the model’s forward pass (e.g., generating gating tensors for the recurrent attention). `torch.rand` without a `generator=` argument uses the global default RNG, which is **not** synchronized across TP ranks. Consequently, each rank will generate different `g` and `beta` tensors, breaking the deterministic behavior required by LLaDA’s block‑diffusion attention and leading to divergent outputs across TP workers.

No `torch.tril` or `kv_cache.append` usage appears in this snippet, so the other checks are not applicable. The fix is to pass a `generator` seeded from a TP‑synchronized seed (e.g., `torch.Generator().manual_seed(sync_seed)`) to every `torch.rand`/`torch.randn` call used in the model’s forward path.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/fused_recurrent.py` : 281
```python
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))

```
> [CONFIRMED BUG]

**Why:**  
The snippet uses `torch.rand(B, T, HV, device='cuda')` (and similarly `torch.randn`) without supplying a `generator` argument. In Tensor‑Parallel (TP) mode each GPU rank invokes this code independently, so each rank will draw a different sequence of random numbers from its own default generator state. LLaDA’s block‑diffusion algorithm requires *identical* random samples (e.g., for noise, dropout masks, β‑schedule, etc.) across all TP partitions to keep the recurrent state and the diffusion trajectory synchronized. Without a TP‑synchronized seed passed via `generator=torch.Generator().manual_seed(tp_seed)`, the randomness is unsynchronized, leading to divergent attention outputs and broken diffusion consistency.

* No `torch.tril` appears in the excerpt, so there is no mask‑related issue.  
* No `kv_cache.append` is shown, thus not relevant to overwrite semantics.

**Fix recommendation:**  
Introduce a TP‑wide random generator (e.g., broadcast a seed from rank 0 and create a `torch.Generator` on each rank) and pass it to all stochastic calls:

```python
# Example TP‑synchronized generator
if torch.distributed.is_initialized():
    # rank 0 decides the seed, others receive it
    seed = torch.tensor(0, dtype=torch.int64, device='cuda')
    if torch.distributed.get_rank() == 0:
        seed = torch.randint(0, 2**31, (1,), device='cuda')
    torch.distributed.broadcast(seed, src=0)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed.item())
else:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(42)  # fallback for non‑TP runs

beta = torch.rand(B, T, HV, device='cuda', generator=gen).sigmoid()
# similarly for other torch.rand / torch.randn calls
```

Ensuring all TP ranks share the same generator guarantees deterministic, synchronized random sampling, preserving LLaDA’s diffusion dynamics.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/fused_recurrent.py` : 282
```python
    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected

```
> [CONFIRMED BUG]

The example code creates all stochastic tensors (`torch.randn`, `torch.rand`, `torch.logsigmoid(torch.rand(...))`, `torch.rand(...).sigmoid()`) **without** supplying a `generator` argument tied to a tensor‑parallel (TP) synchronized seed. In a TP deployment each rank must draw identical random values for any tensor that is later shared (e.g., when constructing the initial recurrent state `h0` or the gating tensors `g`/`beta`).  

Because the calls rely on the default global RNG, each TP rank will generate different values, leading to divergent forward passes and breaking the deterministic behavior required by LLaDA’s block‑diffusion (bidirectional attention) pipeline.  

No `torch.tril` or `kv_cache.append` usage appears in the shown snippet, so those concerns are not applicable here. The core issue is the unsynchronized random sampling, which must be fixed by passing a TP‑wide `torch.Generator` (or using `torch.manual_seed` coordinated across ranks) to every random call.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/fused_recurrent.py` : 283
```python
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)

```
> [CONFIRMED BUG]

**Why:**  
In the snippet the random tensors are generated with plain calls to `torch.randn` and `torch.rand`:

```python
q      = torch.randn(B, T, H, K, device='cuda')
k      = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
v      = torch.randn(B, T, HV, V, device='cuda')
g      = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
beta   = torch.rand(B, T, HV, device='cuda').sigmoid()
h0     = torch.randn(B, HV, K, V, device='cuda')
```

When running under **Tensor Parallel (TP)** each TP rank must produce *identical* random numbers for the parts of the computation that are meant to be shared (e.g., the initial state `h0`, the gating tensors `g` and `beta`, etc.). The standard `torch.rand`/`torch.randn` APIs use the global RNG state, which is *not* automatically synchronized across TP ranks.  

The code does **not** pass a `generator` argument that would be derived from a TP‑synchronized seed, nor does it set the RNG state manually before each call. Consequently each rank will sample different values, breaking the deterministic behavior required by LLaDA’s block‑diffusion process and causing divergence between TP shards.

*The other concerns do not apply to this snippet:*  
- No use of `torch.tril`, so there’s no mask‑related issue.  
- No `kv_cache.append` operation is shown, so there’s no overwrite problem in a diffusion loop.  

**Fix Recommendation:**  
Introduce a TP‑wide seeded generator (e.g., `torch.Generator().manual_seed(tp_global_seed)`) and pass it explicitly to every random call, or synchronize the RNG state across ranks before sampling. This guarantees that all TP ranks see the same random tensors, preserving LLaDA’s correctness.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/fused_recurrent.py` : 284
```python
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(

```
> [CONFIRMED BUG]

The snippet creates the tensors `q`, `k`, `v`, `g`, and `beta` with **`torch.rand`** (or `torch.randn`) **without supplying a `generator` argument**. In a Tensor‑Parallel (TP) setting each replica must draw identical random numbers from a **TP‑synchronised seed**; otherwise each rank will generate different values, breaking the required determinism for the Block‑Diffusion / LLaDA algorithm.

- No `generator=` is passed, so each TP rank will use its own default RNG state.
- Consequently the attention values (`q`, `k`, `v`, `g`, `beta`) diverge across ranks, leading to inconsistent forward passes and invalid diffusion steps.

The other checks (`torch.tril` for masks, `kv_cache.append` handling) are not present in this snippet, so they are not applicable. The core issue is the unsynchronised random sampling. The implementation should use a TP‑wide seed and pass a `torch.Generator` (or `torch.manual_seed` on a shared generator) to all calls to `torch.rand`/`torch.randn` in the model code.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/chunk.py` : 169
```python
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )

```
> [CONFIRMED BUG]

The example generates `beta` and `g` with plain `torch.rand` calls:

```python
beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
g    = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
```

These calls **do not pass a `generator` argument**. In Tensor‑Parallel (TP) mode, each replica would draw its own random numbers from its default RNG state, leading to *different* `beta`/`g` tensors across TP ranks. For LLaDA's Block Diffusion the forward pass must be **deterministic and identical** across all TP shards; otherwise the bidirectional attention within blocks diverges and the diffusion process becomes inconsistent.

Since the code lacks a TP‑synchronized seed (e.g., `torch.rand(..., generator=tp_sync_generator)`), the random sampling is unsynchronized and constitutes a correctness bug. No `torch.tril` or `kv_cache.append` usage appears in the snippet, so those concerns are moot here.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/chunk.py` : 170
```python
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required

```
> [CONFIRMED BUG]

The example creates `beta` and `g` with `torch.rand` (and subsequently `torch.sigmoid` / `torch.logsigmoid`) **without** supplying a `generator=` argument. In a Tensor‑Parallel (TP) deployment each rank must draw identical random numbers for any tensor that influences the model’s forward pass, otherwise the parallel replicas diverge.  

Because `torch.rand` falls back to the default global RNG, each TP rank will generate different `beta` and `g` values unless the caller manually synchronises the seed across ranks. The snippet does **not** obtain a `torch.Generator` seeded from a TP‑synchronized source, nor does it set the same manual seed before the call. Consequently, the random sampling is unsynchronized and will break consistency for LLaDA’s bidirectional block diffusion when TP is enabled.  

(There is no `torch.tril` or `kv_cache.append` in the shown code, so those checks are not applicable here.)

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/chunk.py` : 171
```python

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))

```
> [CONFIRMED BUG]  

The example uses `torch.rand` to generate `beta` and `g`:

```python
beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
g    = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
```

No `generator=` argument is supplied, so the random numbers are drawn from the default global RNG on each GPU. In Tensor‑Parallel (TP) mode this leads to **unsynchronized random seeds** across the parallel ranks, causing each rank to receive a different `beta`/`g` tensor. LLaDA’s block‑diffusion algorithm assumes identical stochastic inputs across TP ranks; mismatched values will break the bidirectional attention consistency and degrade model quality.

* **Fix recommendation** – pass a TP‑synchronized `torch.Generator` (derived from a shared seed) to `torch.rand`, e.g.:

```python
rng = torch.Generator(device='cuda')
rng.manual_seed(shared_seed)          # seed broadcasted across TP ranks
beta = torch.rand(B, T, H, dtype=torch.bfloat16,
                  device='cuda', generator=rng).sigmoid()
g    = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16,
                               device='cuda', generator=rng))
```

or generate the random tensors once on a master rank and broadcast them to all TP ranks before use. This ensures deterministic, rank‑aligned sampling and preserves LLaDA’s diffusion consistency.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/chunk.py` : 172
```python
    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected

```
> [CONFIRMED BUG]

The example uses `torch.rand` to generate `beta` and `g`:

```python
beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
g    = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
```

In a Tensor‑Parallel (TP) setting each rank must see **identical** random values for these tensors, otherwise the forward passes diverge and the bidirectional attention within LLaDA’s block diffusion becomes inconsistent across ranks. The call does **not** pass a `generator=` argument, nor does it derive the seed from a TP‑synchronised source (e.g., `torch.distributed.get_rank()`‑based seed or a common `torch.Generator`). Consequently each TP replica will generate its own independent random numbers, breaking TP consistency.

* No `torch.tril` or `kv_cache.append` usage appears in the snippet, so those concerns are not relevant here. The primary issue is the unsynchronised `torch.rand` calls.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/chunk.py` : 173
```python
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)

```
> [CONFIRMED BUG]  

The snippet generates `beta` and `g` with plain `torch.rand` calls:

```python
beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
g    = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
```

In a Tensor‑Parallel (TP) setting each rank must see **identical** random tensors for any operation that influences the model’s forward pass, otherwise the parallel views diverge and the block‑diffusion (LLaDA) computation becomes nondeterministic across ranks.  

- **Missing generator argument**: `torch.rand` supports a `generator=` kwarg that can be seeded once (e.g., from a TP‑synchronized seed) and passed to every rank. The code does **not** supply such a generator, so each rank will draw independent random numbers.  
- **Impact on LLaDA**: `beta` and `g` are used in the gated‑delta‑rule attention computation. Any mismatch in these tensors across TP ranks will break the required bidirectional consistency inside each diffusion block, leading to incorrect attention outputs and training instability.  

There is no `torch.tril` or `kv_cache.append` in the shown snippet, so those checks are irrelevant here.  

**Fix recommendation**:  
Introduce a TP‑wide random generator (e.g., `torch.Generator().manual_seed(tp_global_seed)`) and pass it to every `torch.rand` call:

```python
tp_generator = torch.Generator(device='cuda')
tp_generator.manual_seed(shared_tp_seed)

beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda',
                  generator=tp_generator).sigmoid()
g    = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16,
                               device='cuda', generator=tp_generator))
```

Ensuring synchronized sampling eliminates the inconsistency bug in TP mode.

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/fla/chunk.py` : 174
```python
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(

```
> [CONFIRMED BUG]  
The snippet uses `torch.rand` to generate `beta` and `g` without supplying a `generator` argument. In Tensor‑Parallel (TP) mode each rank will call `torch.rand` independently, producing different random tensors on each device. LLaDA’s block‑diffusion algorithm requires **exactly the same** gating parameters across all TP slices; otherwise the bidirectional attention within a block becomes inconsistent and the diffusion process diverges.  

Because no TP‑synchronised seed (e.g., a `torch.Generator` created from a common seed and passed via the `generator=` kwarg) is provided, the random sampling is unsynchronised. This is a concrete bug that can cause nondeterministic behavior and incorrect model outputs when TP is enabled.  

*No `torch.tril` or `kv_cache.append` usage is shown in the snippet, so those checks are irrelevant here.*

---
## 🚨 UNSYNC_RANDOM in `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` : 271
```python
        (1, 64),
        (64, 64),
        (128, 64),
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64),
    ]:
        dim_nope_and_rope = 512 + 64

        input_k_cache = torch.randn(
            (num_blocks, block_size, 1, dim_nope_and_rope),
            dtype=torch.bfloat16,
            device="cuda",
        )

        actual_quant = _quantize_k_cache_fast_wrapped(input_k_cache)

        page_table_1 = torch.arange(
            num_blocks * block_size, dtype=torch.int32, device="cuda"
        )

```
> [CONFIRMED BUG]  

The snippet creates `input_k_cache` with:

```python
input_k_cache = torch.randn(
    (num_blocks, block_size, 1, dim_nope_and_rope),
    dtype=torch.bfloat16,
    device="cuda",
)
```

* **Unsynchronized randomness** – `torch.randn` (just like `torch.rand`) draws values from the default global RNG on each GPU. In Tensor‑Parallel (TP) mode each rank will invoke this call independently, producing *different* random tensors unless a common `generator` (seeded with a TP‑synchronised value) is supplied.  
* **Impact on LLaDA** – LLaDA’s block‑diffusion / bidirectional‑attention logic assumes that the key‑cache (`k_cache`) is *identical* across all TP ranks for the same logical block. Divergent caches break the deterministic behavior required for diffusion steps and can lead to inconsistent attention scores, corrupting the diffusion process.  
* **Missing fix** – The code does not pass a `generator=` argument, nor does it set the RNG state from a TP‑wide seed before the call. Consequently the random initialization of the K‑cache is not reproducible nor synchronized across ranks.  

**Recommendation**: Replace the call with a TP‑synchronised generator, e.g.:

```python
tp_seed = get_tp_sync_seed()          # a seed broadcasted/derived from rank‑0
gen = torch.Generator(device="cuda").manual_seed(tp_seed)
input_k_cache = torch.randn(
    (num_blocks, block_size, 1, dim_nope_and_rope),
    dtype=torch.bfloat16,
    device="cuda",
    generator=gen,
)
```

or explicitly set the RNG state on each rank before the call. This ensures that every TP replica receives the same initial `k_cache`, preserving LLaDA’s diffusion correctness.

---
