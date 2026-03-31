# Diffusion CUDA Graph v2 — Step-Level Piecewise Design Spec

*Revision 4: Addresses review round 3 feedback (I10-I12, M4-M5).*

## Problem

The existing `DiffusionCudaGraphRunner` captures the entire `dit.forward()` as a single CUDA Graph. This works for the simple case (no cache-dit), but is **incompatible with cache-dit** because:

- Cache-dit (DBCache) needs to make a per-step decision after executing the first Fn blocks: should the remaining blocks use cached outputs or full compute?
- A single CUDA Graph captures one fixed execution path — it cannot branch at replay time.

Additionally, the current implementation lacks:

1. **Bucketing**: No support for multiple resolutions; each new resolution triggers a fresh capture.
2. **Warmup**: No pre-capture of common resolutions at startup; first request per resolution pays capture latency.

## Goals

1. **Cache-dit compatibility**: Support SCM (step-level mask) and DBCache (step-level dynamic decision) with CUDA Graph acceleration.
2. **Bucketing with warmup**: Pre-capture graphs for configurable resolution set at startup; lazy capture for unseen resolutions.
3. **Backward compatibility**: Preserve the existing single-graph path as a fallback when cache-dit is disabled.

## Scope

| Constraint | Value | Reason |
|---|---|---|
| Model | ZImage-Turbo | Primary optimization target; other models deferred |
| Cache-dit modes | SCM + DBCache (step-level) | Per-block (PrunedBlocks) deferred to future evaluation |
| Attention | FlashAttention | No timestep-dependent kernel paths |
| CFG | Disabled | ZImage: `should_use_guidance=False` |
| Model switching | Disabled | ZImage: `boundary_ratio=None` |

## Architecture

### Core Idea: Step-Level Graph Splitting

Instead of capturing the entire `dit.forward()` as one graph, split it into segments with an **eager decision point** between the Fn blocks and the remaining blocks:

```
dit.forward() split into segments:

  Graph: PRE-BLOCK
  ┌──────────────────────────────────────────────────────┐
  │ timestep_embed -> patchify -> noise_refiner x2 ->    │
  │ cap_embed -> context_refiner x2 -> concat unified    │
  └──────────────────────────────────────────────────────┘
                         |
  Graph: FN-BLOCKS (first Fn=1 block, always full compute)
  ┌──────────────────────────────────────────────────────┐
  │ layers[0].forward(unified, ...)                      │
  └──────────────────────────────────────────────────────┘
                         |
  [EAGER] DBCache decision: can_cache(Fn_output) -> bool
                         |
              ┌──────────┴──────────┐
              v                     v
  Graph: MIDDLE-FULL          [EAGER] copy cached_outputs
  ┌────────────────────┐
  │ layers[1..N-Bn-1]  │
  │ full compute       │
  └────────────────────┘
              └──────────┬──────────┘
                         |
  Graph: BN-BLOCKS (last Bn blocks, always full compute)
  ┌──────────────────────────────────────────────────────┐
  │ layers[N-Bn..N-1].forward(unified, ...)              │
  │ (NOT captured when Bn=0; skipped entirely)           │
  └──────────────────────────────────────────────────────┘
                         |
  Graph: POST-BLOCK
  ┌──────────────────────────────────────────────────────┐
  │ final_layer -> unpatchify -> output                  │
  └──────────────────────────────────────────────────────┘
```

### Graph Inventory Per Resolution

| Graph | Content | When replayed |
|-------|---------|---------------|
| `graph_pre` | Pre-block (patch embed, noise/context refiners, concat) | Every step |
| `graph_fn` | First Fn blocks (full compute) | Every compute step (skipped on SCM all-cache) |
| `graph_middle_full` | Blocks Fn..N-Bn-1 (full compute) | When DBCache decides compute |
| `graph_bn` | Last Bn blocks (full compute) | Every compute step **only if Bn > 0**; not captured/replayed when Bn=0 |
| `graph_post` | Post-block (final layer, unpatchify) | Every step |

With default Fn=1, Bn=0: **3 active graphs per resolution** (graph_pre, graph_fn, graph_middle_full) + graph_post = **4 total**. `graph_bn` is neither captured nor replayed.

### Fallback: Whole-Graph Mode (No Cache-Dit)

When cache-dit is disabled, the existing `DiffusionCudaGraphRunner` is used. It captures the entire `dit.forward()` as a single graph — maximum performance, no decision overhead.

| Scenario | Runner | Graphs |
|----------|--------|--------|
| CUDA Graph ON + cache-dit OFF | `DiffusionCudaGraphRunner` (existing) | 1 whole graph |
| CUDA Graph ON + cache-dit ON | `StepLevelCudaGraphRunner` (new) | 4-5 graphs |
| CUDA Graph OFF | None (eager) | 0 |

## Execution Flow Per Denoising Step

```
with set_forward_context(current_timestep=i, attn_metadata=None, forward_batch=batch):
  # set_forward_context wraps the ENTIRE step. FlashAttention does not
  # read current_timestep, but the context is maintained for profiling,
  # logging, and any code that reads the forward context.
  # All graph replays within this step share the same context.

  for each denoising step i:

    SCM decision (pre-computed): is_compute_step = scm_mask[i]

    if not is_compute_step:
        === SCM all-cache fast path ===
        graph_pre.replay()
        inter_buffer.copy_(cached_outputs)    [eager, single copy]
        graph_post.replay()

        Note: graph_pre produces adaln_input (needed by graph_post's
        final_layer) so it cannot be skipped entirely. The noise_refiner
        and context_refiner computation within graph_pre is wasted in
        this path. See "SCM Cache Step Overhead Analysis" section below.

    elif cache_dit_enabled:
        === DBCache decision path ===
        graph_pre.replay()
        graph_fn.replay()

        [EAGER] step_can_cache = context_manager.can_cache(inter_buffer)
                (GPU similarity calc + .item() sync + state update)

        if step_can_cache:
            inter_buffer.copy_(cached_outputs)    [eager copy]
        else:
            graph_middle_full.replay()
            cached_outputs.copy_(inter_buffer)    [update cache]

        if self.bn_blocks > 0:                    [conditional, not called when Bn=0]
            graph_bn.replay()

        graph_post.replay()

    else:
        === No cache-dit (should not reach here; whole-graph used instead) ===
        graph_all_full.replay()
        graph_post.replay()

    scheduler.step(noise_pred, t, latents)        [always outside graph]
```

### SCM Cache Step Overhead Analysis

On SCM cache steps, `graph_pre` is still replayed because `graph_post` requires `adaln_input` (produced by `graph_pre`'s timestep embedding). The noise_refiner (2 blocks) and context_refiner (2 blocks) computations are wasted.

**Quantification**: Pre-block contains 4 refiner blocks + patch embed + linear proj + concat. The 30 main transformer blocks dominate computation. Importantly, the refiner blocks operate on **shorter sequences** than main blocks: noise_refiner processes only image tokens (seq_len=256 for 256×256), while main blocks process the full unified sequence (image + caption tokens, ~512). Additionally, context_refiner blocks have `modulation=False` (no adaLN), further reducing their compute. Pre-block overhead is estimated at **< 2% of total step time**. This waste is acceptable for the initial implementation. A future optimization could split `graph_pre` into `graph_pre_timestep` (just timestep embed → adaln_input) and `graph_pre_full`, replaying only `graph_pre_timestep` on cache steps.

### dit.forward() Return Type and Post-Processing

ZImage's `forward()` returns a single `torch.Tensor`: `return -x[0]` (shape `[B, C, H, W]`).

In the eager path, `_predict_noise_with_cfg()` applies two post-processing steps after `dit.forward()`:

1. **`slice_noise_pred(noise_pred, latents)`** — ZImage inherits the base implementation which is a no-op (`return noise`).
2. **CFG combine** — ZImage has `should_use_guidance=False`, so `do_classifier_free_guidance` is `False` and the function returns immediately after the positive pass.

Both steps are no-ops for ZImage. The graph path safely skips them — the `forward_post_block()` output (`-x[0]`) is the final `noise_pred` with no further transformation needed.

### GPU-to-CPU Synchronization

Per denoising step: **1 sync** (the `.item()` call inside `can_cache()`), only on compute steps.

Compared to eager mode: no change in sync count (cache-dit eager also does 1 sync per step). The improvement is that all GPU kernels within each segment are launched via a single `cudaGraphLaunch()` instead of individually.

## Cache-Dit State Machine Compatibility

### Problem: Bypassing CachedBlock Wrapper (Review C1)

Cache-dit's `CachedBlock` wrapper maintains internal state (cached output tensors, step counter, residual diff accumulation). If we bypass the wrapper via `_get_raw_block()` and call raw blocks directly, the wrapper's state is never updated, causing `can_cache()` to malfunction.

### Solution: Explicit State Synchronization

The segmented forward methods bypass `CachedBlock.forward()` for GPU computation (which is captured in graphs), but we **explicitly synchronize cache-dit state** at the eager decision point:

```python
def replay_step(self, ...):
    ...
    graph_fn.replay()

    # === EAGER: Cache-dit state synchronization ===
    # After Fn blocks replay, inter_buffer contains the Fn output.
    # We must update cache-dit's internal state to match what
    # CachedBlock.forward() would have done:
    #
    # 1. Call context_manager.mark_step_begin() at step start
    #    (already called by cache-dit's step-level hook in refresh_context)
    #
    # 2. Feed the Fn output to can_cache() for similarity computation.
    #    can_cache() internally:
    #    - Computes similarity(prev_fn_output, current_fn_output)
    #    - Calls add_residual_diff() to update accumulated stats
    #    - Updates consecutive_cache_count
    #    - Returns the decision
    #
    # 3. If full compute: after graph_middle_full.replay(),
    #    update CachedBlock wrappers' cached output tensors so they
    #    have correct data for the next step's similarity comparison.

    step_can_cache = self._query_dbcache_decision()

    if step_can_cache:
        inter_buffer.copy_(cached_outputs)
    else:
        graph_middle_full.replay()
        cached_outputs.copy_(inter_buffer)
        # Update the Fn wrapper's cached output for next step's
        # similarity baseline:
        self._update_fn_wrapper_cached_output()
    ...
```

### `_update_fn_wrapper_cached_output()` Implementation (Review I6)

After a full-compute step, the CachedBlock wrappers' internal `cached_output` tensors must be updated so that next step's `can_cache()` similarity comparison has a valid baseline.

```python
def _update_fn_wrapper_cached_output(self):
    """Update CachedBlock wrappers' cached outputs after a full-compute step.

    Which blocks need updating:
    - ALL blocks (Fn + middle + Bn) that were executed in full compute.
      Each CachedBlock wrapper stores its own block's output as the
      baseline for similarity comparison. However, cache-dit's step-level
      CachedBlocks pattern compares only at the Fn boundary — the context
      manager uses the Fn block output (not per-block outputs) for
      similarity.

    What to update:
    - The context_manager's internal "previous output" reference, which
      is the tensor that can_cache() compares against in the next step.
    - This is typically stored as context_manager._cached_residual or
      similar attribute.

    Data source:
    - For Fn blocks: the inter_buffer state right after graph_fn.replay()
      (saved in self._fn_output_snapshot before the middle decision).
    - For all blocks: the inter_buffer state after graph_middle_full.replay()
      (current self.inter_buffer).

    Implementation:
    - The exact attribute name must be confirmed by inspecting cache-dit's
      CachedContextManager source during the eager-mode prototype phase.
    - The update is a simple tensor.copy_() to the wrapper's internal buffer.
    """
    # Pseudocode — exact API to be confirmed in eager prototype:
    ctx_mgr = self.transformer._context_manager
    # Update the similarity baseline with current Fn output
    if hasattr(ctx_mgr, '_cached_residual'):
        ctx_mgr._cached_residual.copy_(self._fn_output_snapshot)
    # Alternative: ctx_mgr.update_cached_output(self._fn_output_snapshot)
```

### `_query_dbcache_decision()` Implementation (Review I7)

```python
def _query_dbcache_decision(self) -> bool:
    """Call cache-dit's can_cache() in eager mode after Fn blocks replay.

    Parameter semantics:
    - can_cache() expects the Fn block output tensor — specifically the
      full unified tensor (image + caption tokens) after the first Fn
      blocks have executed. This is the content of inter_buffer right
      after graph_fn.replay().
    - cache-dit's similarity() computes: diff = mean(|t1 - t2|) / mean(|t1|)
      where t1 = current Fn output, t2 = previous step's Fn output.
      The comparison operates on the entire unified tensor (not just
      image tokens), because the Fn block processes the full concatenated
      sequence.

    Distributed group handling:
    - can_cache() internally calls similarity(), which in SP/TP mode
      performs all_reduce via the groups set on context_manager by
      _patch_cache_dit_similarity() and _mark_transformer_parallelized().
    - These groups (_sglang_sp_group, _sglang_tp_group, _sglang_tp_sp_group)
      are set once during enable_cache_on_transformer() and persist on
      the context_manager object. They are NOT affected by segmented
      forward — the groups are properties of the model, not the forward
      call path.
    - Therefore, distributed decisions remain consistent in piecewise mode.

    Returns True if remaining blocks can use cache (skip computation).
    """
    # Save Fn output snapshot for _update_fn_wrapper_cached_output()
    # Use pre-allocated buffer (allocated at capture time) + copy_ instead
    # of clone() to avoid per-step memory allocation overhead (~1-2us).
    # The copy itself (~8-16us for 1024x1024) is unavoidable since
    # inter_buffer will be overwritten by subsequent graph replays.
    self._fn_output_snapshot.copy_(self.inter_buffer)

    ctx_mgr = self.transformer._context_manager
    return ctx_mgr.can_cache(self.inter_buffer)
```

### Pre-Implementation Validation

Before implementing the full CUDA Graph integration, an **eager-mode prototype** must be written to validate cache-dit state consistency:

1. Run the segmented forward methods (pre → fn → middle → bn → post) in eager mode (no graphs)
2. Compare the per-step `can_cache()` decisions against the baseline (original `CachedBlock.forward()` path)
3. Verify that all DBCache internal states (residual_diff, consecutive_cache_count, etc.) match

This prototype costs minimal effort (call the segmented methods sequentially without capture) and eliminates the highest-risk correctness concern before investing in graph capture logic.

## StepLevelCudaGraphRunner Class Design

Location: `runtime/managers/diffusion_cuda_graph_runner.py` (alongside existing `DiffusionCudaGraphRunner`)

```python
class StepLevelCudaGraphRunner:
    """Step-level piecewise CUDA Graph runner for cache-dit compatible inference.

    Captures dit.forward() as multiple segments with an eager decision point
    between Fn blocks and remaining blocks, enabling DBCache step-level
    decisions at replay time.
    """

    def __init__(self, device, num_blocks, fn_blocks, bn_blocks, pool):
        self.device = device
        self.num_blocks = num_blocks
        self.fn_blocks = fn_blocks        # from SGLANG_CACHE_DIT_FN
        self.bn_blocks = bn_blocks        # from SGLANG_CACHE_DIT_BN
        self.pool = pool                  # shared across ALL runners (both types)

        # Graphs
        self.graph_pre: Optional[CUDAGraph] = None
        self.graph_fn: Optional[CUDAGraph] = None
        self.graph_middle_full: Optional[CUDAGraph] = None
        self.graph_bn: Optional[CUDAGraph] = None   # None when bn_blocks == 0
        self.graph_post: Optional[CUDAGraph] = None

        # Fixed-address buffers
        self.input_buffers: dict = {}       # timestep, latents
        self.inter_buffer: Tensor = None    # hidden_states between segments
        self.output_buffer: Tensor = None   # final noise_pred
        self.cached_outputs: Tensor = None  # middle segment cache for DBCache

        # State buffers shared across segments
        self.adaln_input_buffer: Tensor = None
        self.unified_freqs_cis_buffer: tuple = None

        self._captured = False

    def capture(self, dit_model, timestep, latents, static_kwargs):
        """Capture all segments. Called once per resolution.

        Must be called within a set_forward_context() block to ensure
        warmup runs and the capture run share the same forward context.
        The context is read by attention backends and profiling code
        during the warmup eager executions.

        Captures: graph_pre, graph_fn, graph_middle_full, graph_post.
        Captures graph_bn only if self.bn_blocks > 0.
        """
        ...

    def replay_step(self, timestep, latents, step_is_all_cached, cache_dit_ctx_mgr):
        """Replay one denoising step with step-level cache-dit decision."""
        ...

    def update_static_kwargs(self, static_kwargs):
        """Update per-request inputs (encoder_hidden_states etc.).

        Called at the start of each new request to update tensor data
        that changes per-request (e.g., prompt embeddings) while keeping
        the same fixed addresses.
        """
        ...

    def reset(self):
        """Release all graphs and buffers."""
        ...
```

## ZImage DiT Model Modifications

Add segmented forward methods to `ZImageTransformer2DModel`:

```python
@dataclass
class PreBlockOutput:
    """Output of forward_pre_block, separating GPU tensors from metadata.

    GPU tensors (addresses matter for graph capture):
        unified: Hidden states after pre-block processing.
        unified_freqs_cis: RoPE embeddings (constant across steps for same resolution).
        adaln_input: Timestep conditioning (updated each step by graph_pre).

    Metadata (Python values, fixed per resolution, stored on runner at capture time):
        x_size: Spatial dimensions for unpatchify.
        x_local_seq_len: Local sequence length (for SP restore in post_block).
        use_full_unified_sequence: Whether to use full unified sequence (SP mode).
        num_replicated_suffix: Number of replicated suffix tokens (caption tokens in non-SP mode).
    """
    unified: torch.Tensor
    unified_freqs_cis: Tuple[torch.Tensor, torch.Tensor]
    adaln_input: torch.Tensor
    x_size: list
    x_local_seq_len: int
    use_full_unified_sequence: bool
    num_replicated_suffix: int

class ZImageTransformer2DModel:
    # Existing forward() preserved unchanged (eager + whole-graph fallback)

    def forward_pre_block(self, hidden_states, encoder_hidden_states,
                          timestep, guidance, patch_size, f_patch_size,
                          freqs_cis) -> PreBlockOutput:
        """Timestep embed -> patchify -> noise_refiner x2 ->
        cap_embed -> context_refiner x2 -> concat unified.

        Input types:
            hidden_states: List[torch.Tensor] — ZImage wraps a single
                [C, H, W] tensor in a list. Same type as forward().
            encoder_hidden_states: List[torch.Tensor] — prompt embeddings.

        Note on adaln_input computation:
            The original code does `adaln_input = t.type_as(x)` where
            x = hidden_states (a List). This works because type_as()
            is never actually called on a list — the timestep embedder
            output `t` already has the correct dtype from the model's
            weight dtype. The implementation should use explicit dtype
            casting: `adaln_input = t.to(dtype=target_dtype)` where
            target_dtype is determined from model weights, avoiding the
            ambiguous type_as(list) call.

        CUDA Graph compatibility note on conditional padding:
            The original forward() contains conditional padding:
                if x_valid_lens[0] < x.shape[0]:
                    x[x_valid_lens[0]:] = self.x_pad_token
            During capture, the Python `if` branch is evaluated once and
            the resulting kernel sequence is baked in. To ensure the
            padding kernel is always captured, the implementation should
            either: (a) ensure warmup dummy data triggers the padding
            branch, or (b) make padding unconditional (always execute
            the slice assignment — when valid_lens == shape[0], the slice
            is empty and the kernel writes 0 elements, which is safe).
            Option (b) is recommended for robustness.

        Returns: PreBlockOutput dataclass.
        """
        ...

    def forward_fn_blocks(self, unified, unified_freqs_cis, adaln_input,
                          num_replicated_suffix, fn_count):
        """Execute first fn_count main transformer blocks (always full compute).

        Bypasses CachedBlock wrappers and calls raw ZImageTransformerBlock
        directly. Cache-dit state synchronization is handled externally
        by StepLevelCudaGraphRunner (see "Cache-Dit State Machine
        Compatibility" section).

        Args:
            fn_count: Number of Fn blocks to execute (from SGLANG_CACHE_DIT_FN).

        Returns: updated unified tensor.
        """
        for i in range(fn_count):
            raw_block = self._get_raw_block(i)
            unified = raw_block(unified, unified_freqs_cis, adaln_input,
                                num_replicated_suffix=num_replicated_suffix)
        return unified

    def forward_middle_blocks(self, unified, unified_freqs_cis, adaln_input,
                              num_replicated_suffix, fn_count, bn_count):
        """Execute middle blocks (Fn..N-Bn-1) in full compute mode.

        Returns: updated unified tensor.
        """
        end = len(self.layers) - bn_count
        for i in range(fn_count, end):
            raw_block = self._get_raw_block(i)
            unified = raw_block(unified, unified_freqs_cis, adaln_input,
                                num_replicated_suffix=num_replicated_suffix)
        return unified

    def forward_bn_blocks(self, unified, unified_freqs_cis, adaln_input,
                          num_replicated_suffix, bn_count):
        """Execute last bn_count blocks (always full compute).

        Not called when bn_count=0 (default). The caller is responsible
        for skipping this method when bn_count=0; no graph is captured
        for this segment in that case.

        Returns: updated unified tensor.
        """
        if bn_count == 0:
            return unified
        start = len(self.layers) - bn_count
        for i in range(start, len(self.layers)):
            raw_block = self._get_raw_block(i)
            unified = raw_block(unified, unified_freqs_cis, adaln_input,
                                num_replicated_suffix=num_replicated_suffix)
        return unified

    def forward_post_block(self, unified, adaln_input, patch_size,
                           f_patch_size, x_size, x_local_seq_len,
                           use_full_unified_sequence):
        """Final layer -> SP restore -> unpatchify -> output.

        CUDA Graph compatibility note (Review I9):
        This method contains Python-level operations that are "baked in"
        during capture:
        - `list(unified.unbind(dim=0))` — Python list creation from tensor
        - `self.unpatchify(x, ...)` — iterates over list, applies view/permute/reshape
        - `return -x[0]` — Python indexing

        These are safe for CUDA Graph because:
        1. Shape is fixed for a given resolution, so unbind/list/indexing
           produce the same tensor count and shapes on every replay.
        2. view/permute/reshape are metadata-only ops (no new allocation)
           when possible, or produce allocations that 2x warmup stabilizes.
        3. The 2x warmup before capture ensures PyTorch's caching allocator
           is in steady state — all internal allocations from reshape ops
           are pre-warmed.

        Returns: noise_pred tensor (single torch.Tensor, shape [B, C, H, W]).
        """
        ...

    def _get_raw_block(self, block_idx):
        """Get the underlying ZImageTransformerBlock, bypassing CachedBlock wrapper.

        WARNING: Bypassing CachedBlock means cache-dit's internal state
        is NOT updated by this call. The caller (StepLevelCudaGraphRunner)
        must explicitly synchronize cache-dit state — see "Cache-Dit State
        Machine Compatibility" section in the design doc.
        """
        layer = self.layers[block_idx]
        if hasattr(layer, "transformer_blocks"):
            return layer.transformer_blocks[0]
        return layer
```

## Inter-Buffer Address Sharing

All segments share a single fixed-address `inter_buffer` for hidden_states:

```
graph_pre output    -> inter_buffer (fixed address)
graph_fn input      <- inter_buffer
graph_fn output     -> inter_buffer (same address, in-place update)
graph_middle input  <- inter_buffer
graph_middle output -> inter_buffer
graph_bn input      <- inter_buffer  (only if Bn > 0)
graph_bn output     -> inter_buffer  (only if Bn > 0)
graph_post input    <- inter_buffer
graph_post output   -> output_buffer (fixed address)
```

The `adaln_input` buffer is produced by `graph_pre` and consumed by all block segments and `graph_post` (final_layer uses adaln_input for adaLN modulation). Its address is fixed at capture time; `graph_pre` updates its content each step.

`unified_freqs_cis` is resolution-dependent and constant across steps. Its address is bound at capture time.

### Capture Sequence and Address Binding (Review I8)

Segments must be captured in **strict order** (pre → fn → middle → bn → post) because each segment's input address must match the previous segment's output address. The mechanism to ensure address consistency:

```python
def capture(self, dit_model, timestep, latents, static_kwargs):
    # Phase 1: One full eager forward to determine shapes and allocate
    # fixed-address buffers.
    with torch.no_grad():
        (unified, unified_freqs_cis, adaln_input,
         x_size, x_local_seq_len, use_full_unified_sequence,
         num_replicated_suffix) = dit_model.forward_pre_block(
            latents, static_kwargs['encoder_hidden_states'],
            timestep, ...)

    # Allocate fixed-address inter_buffer matching unified's shape
    self.inter_buffer = torch.empty_like(unified)
    self.adaln_input_buffer = torch.empty_like(adaln_input)
    self.cached_outputs = torch.empty_like(unified)

    # Phase 2: Capture each segment sequentially.
    # Each segment's capture function explicitly reads from and writes
    # to self.inter_buffer (same Python object = same GPU address).

    # Capture pre_block: output MUST write into self.inter_buffer
    def pre_fn():
        result = dit_model.forward_pre_block(...)
        # result[0] is unified — copy into fixed-address buffer
        self.inter_buffer.copy_(result[0])
        self.adaln_input_buffer.copy_(result[2])
        return result
    self.graph_pre = self._capture_one(pre_fn)

    # Capture fn_blocks: reads self.inter_buffer, writes self.inter_buffer
    def fn_fn():
        out = dit_model.forward_fn_blocks(
            self.inter_buffer, self.unified_freqs_cis_buffer,
            self.adaln_input_buffer, ...)
        self.inter_buffer.copy_(out)
    self.graph_fn = self._capture_one(fn_fn)

    # Capture middle_blocks: same pattern
    def middle_fn():
        out = dit_model.forward_middle_blocks(
            self.inter_buffer, self.unified_freqs_cis_buffer,
            self.adaln_input_buffer, ...)
        self.inter_buffer.copy_(out)
    self.graph_middle_full = self._capture_one(middle_fn)

    # Capture bn_blocks (only if bn_blocks > 0)
    if self.bn_blocks > 0:
        def bn_fn():
            out = dit_model.forward_bn_blocks(
                self.inter_buffer, self.unified_freqs_cis_buffer,
                self.adaln_input_buffer, ...)
            self.inter_buffer.copy_(out)
        self.graph_bn = self._capture_one(bn_fn)

    # Capture post_block: reads inter_buffer, writes output_buffer
    def post_fn():
        out = dit_model.forward_post_block(
            self.inter_buffer, self.adaln_input_buffer, ...)
        self.output_buffer = out  # address bound at capture
        return out
    self.graph_post = self._capture_one(post_fn)

    self._captured = True
```

**Key invariant**: Every segment capture function uses `self.inter_buffer` as both input and output (via explicit `copy_`). Since `self.inter_buffer` is the same Python object across all captures, its GPU memory address is identical in every graph. This ensures that when segment N's graph is replayed, its output lands at the address that segment N+1's graph expects as input.

**Alternative (in-place)**: If `forward_fn_blocks()` can be made to write its output directly into `inter_buffer` (e.g., via pre-allocated output tensor parameter), the explicit `copy_` can be eliminated. This is a **Phase 2 optimization**: each copy adds ~8us for 1024×1024 (33.4MB / 4TB/s HBM bandwidth), totaling ~25-32us across 3-4 segments. This is measurable (~25-50% of total CPU overhead) but acceptable in Phase 1 since the overall improvement over eager (~12ms) is still ~100×.

## cached_outputs Lifecycle

```
Capture phase:
    cached_outputs = torch.empty_like(inter_buffer)
    (Allocated once, persists across requests for same resolution)

Full compute step (DBCache decides compute):
    graph_middle_full.replay()                # inter_buffer updated
    cached_outputs.copy_(inter_buffer)        # save for future cache steps

Cache step (DBCache decides cache):
    inter_buffer.copy_(cached_outputs)        # restore from cache
    (Single copy, ~5us, not worth capturing as a graph)
```

### Cross-Request Correctness (Review I3)

`cached_outputs` persists across requests (same resolution reuses the runner). Between requests:

1. `refresh_context_on_transformer()` resets cache-dit's internal state (step counter, residual diff, etc.)
2. Cache-dit's `max_warmup_steps` (default=4) guarantees that the **first N steps of every request are compute steps** (warmup check in `can_cache()` returns False during warmup). This means `graph_middle_full` will execute and populate `cached_outputs` with valid data for the current request **before any cache step can occur**.

Therefore, stale `cached_outputs` from the previous request are always overwritten before they could be read. No explicit zeroing is needed.

## Bucketing and Warmup

### Resolution to Latent Shape Mapping (ZImage)

```
Pixel resolution -> latent size -> token count (seq_len)
  spatial_compression_ratio = 8 (FluxVAE)
  PATCH_SIZE = 2

  256x256   -> latent 32x32   -> 16x16  = 256 tokens
  512x512   -> latent 64x64   -> 32x32  = 1024 tokens
  1024x1024 -> latent 128x128 -> 64x64  = 4096 tokens
```

### Graph Cache Structure

```
Graph Cache (dict):
    Key = latent_shape tuple: (B, C, F, H, W)

    (1,16,1,32,32)   -> StepLevelRunner  <- 256x256   (warmup)
    (1,16,1,64,64)   -> StepLevelRunner  <- 512x512   (warmup)
    (1,16,1,128,128) -> StepLevelRunner  <- 1024x1024 (warmup)
    (1,16,1,48,48)   -> StepLevelRunner  <- lazy capture
```

All runners (both `StepLevelCudaGraphRunner` and `DiffusionCudaGraphRunner`) share a single `pool = torch.cuda.graph_pool_handle()` managed by `DenoisingStage`. This ensures memory is shared across all runners regardless of type. The existing `DiffusionCudaGraphRunner` will be updated to accept an external `pool` parameter instead of creating its own.

### Environment Variable

```python
# envs.py
SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES: str = "(256,256),(512,512),(1024,1024)"
```

Usage:

```python
import ast
warmup_sizes = ast.literal_eval(envs.SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES)
# -> [(256,256), (512,512), (1024,1024)]
```

### Warmup Flow

```
Server startup -> Pipeline initialized -> DenoisingStage.warmup():
    pool = torch.cuda.graph_pool_handle()       # single shared pool
    warmup_sizes = ast.literal_eval(envs.SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES)
    for (W, H) in warmup_sizes:
        latent_shape = pipeline_config.compute_latent_shape(W, H)

        # Construct dummy inputs matching expected shapes:
        dummy_latents = torch.randn(latent_shape, device=device, dtype=target_dtype)
        dummy_timestep = torch.zeros(1, device=device, dtype=target_dtype)

        # dummy_static_kwargs construction:
        # - encoder_hidden_states: List[Tensor] with shape matching text encoder
        #   output. Use a fixed caption length (e.g., 256 tokens) since caption
        #   length determines unified seq_len. The warmup capture's graph is only
        #   valid for this specific unified seq_len. If actual requests have
        #   different caption lengths, they will miss the warmup cache and trigger
        #   lazy capture. For ZImage, caption length is typically padded to a
        #   fixed size by the text encoder, so a single warmup per resolution
        #   covers all requests at that resolution.
        # - freqs_cis: Computed from resolution via pipeline_config.get_freqs_cis()
        # - guidance: torch.zeros(1) (ZImage: guidance=0)
        # - patch_size, f_patch_size: from pipeline_config.dit_config
        dummy_static_kwargs = {
            'encoder_hidden_states': [torch.randn(cap_seq_len, hidden_size, ...)],
            'freqs_cis': pipeline_config.compute_freqs_cis(W, H),
            'guidance': torch.zeros(1, device=device),
        }

        if cache_dit_enabled:
            runner = StepLevelCudaGraphRunner(device, num_blocks, fn, bn, pool)
        else:
            runner = DiffusionCudaGraphRunner(device, pool=pool)
        runner.capture(dit_model, dummy_timestep, dummy_latents, dummy_static_kwargs)
        self._runners[latent_shape] = runner
```

### Serving Phase Request Routing

```
Request arrives -> compute latent_shape
    if latent_shape in runners:
        -> hit warmup cache, update per-request kwargs, replay
    else:
        -> lazy capture on first step (using shared pool), cache runner for future reuse
```

**Lazy capture first-request latency**: For resolutions not in the warmup set, the first request triggers capture during its first denoising step. This adds 2× warmup runs + `torch.cuda.synchronize()` + 1× capture run per segment, resulting in approximately **3-4× single step time** of extra latency on that first request only. Subsequent requests with the same resolution replay from cache with no overhead. For latency-sensitive deployments, add all expected resolutions to `SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES` to avoid lazy capture entirely.

## Configuration and Enablement

### Automatic Runner Selection

| `enable_diffusion_cuda_graph` | `SGLANG_CACHE_DIT_ENABLED` | Runner |
|-------------------------------|---------------------------|--------|
| False | * | Eager (no graphs) |
| True | False | `DiffusionCudaGraphRunner` (existing, 1 whole graph) |
| True | True | `StepLevelCudaGraphRunner` (new, 4-5 graphs) |

No new CLI flags needed. The system automatically selects the appropriate runner based on whether cache-dit is enabled.

### Environment Variables Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES` | `"(256,256),(512,512),(1024,1024)"` | Resolutions to pre-capture at startup |
| `SGLANG_CACHE_DIT_FN` | 1 | Number of Fn blocks (always full compute) |
| `SGLANG_CACHE_DIT_BN` | 0 | Number of Bn blocks (always full compute) |

`SGLANG_CACHE_DIT_FN` and `SGLANG_CACHE_DIT_BN` are existing env vars, read by the new runner to determine segment boundaries.

## File Changes

| File | Change | Content |
|------|--------|---------|
| `runtime/managers/diffusion_cuda_graph_runner.py` | New class + modify existing | `StepLevelCudaGraphRunner` (new); `DiffusionCudaGraphRunner` updated to accept external `pool` parameter |
| `runtime/models/dits/zimage.py` | New methods | `forward_pre_block()`, `forward_fn_blocks()`, `forward_middle_blocks()`, `forward_bn_blocks()`, `forward_post_block()`, `_get_raw_block()` |
| `runtime/pipelines_core/stages/denoising.py` | Modify | Warmup logic, shared pool management, runner selection, step-level replay integration in denoising loop, `set_forward_context` wrapping |
| `envs.py` | New var | `SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES` |
| `runtime/server_args.py` | No change | Reuse existing `enable_diffusion_cuda_graph` |

### Files NOT Changed

| File | Reason |
|------|--------|
| `cache_dit_integration.py` | No modification; use `transformer._context_manager.can_cache()` directly |
| cache-dit library | No modification; leverage existing `can_cache()` interface |

## Correctness Guarantees

1. **Capture determinism**: 2x warmup runs before each segment capture to stabilize PyTorch caching allocator. All graphs share `pool=pool` to avoid fragmentation.

2. **Address consistency**: All segments connected through fixed-address `inter_buffer`. Capture binds addresses; replay updates data in-place.

3. **Cache-dit state consistency**: `can_cache()` is called in eager mode between graph replays. Its internal side effects (add_residual_diff, consecutive cache step counting, etc.) execute normally. `refresh_context_on_transformer()` resets statistics at the start of each new request. After full-compute steps, CachedBlock wrapper's internal cached output is explicitly updated to maintain similarity baseline for next step. An eager-mode prototype will validate state consistency before graph capture is implemented (see "Pre-Implementation Validation").

4. **SCM + DBCache interaction**: When SCM mask says cache (mask[i]==0), the entire step skips the DBCache decision and does not call `can_cache()`. This is consistent with cache-dit's internal behavior where SCM cache steps bypass similarity computation entirely. The DBCache internal state (consecutive cache counter, accumulated residual diff, etc.) is not updated on SCM-skipped steps, matching the behavior of the eager CachedBlocks wrapper which also skips these updates when SCM says cache.

5. **Cross-request cached_outputs correctness**: `cached_outputs` may contain stale data from a previous request. This is safe because cache-dit's `max_warmup_steps` (default=4) guarantees the first N steps of every request are compute steps, which will execute `graph_middle_full` and overwrite `cached_outputs` with valid data before any cache step can read them.

6. **Fallback safety**: cache_dit_enabled=False uses existing `DiffusionCudaGraphRunner` (with shared pool). enable_diffusion_cuda_graph=False uses pure eager. Incompatible attention backends trigger assertion errors at startup.

7. **set_forward_context scope**: A single `set_forward_context(current_timestep=i, ...)` wraps the entire step (all graph replays + eager decision). FlashAttention does not read `current_timestep`, but the context is maintained for profiling/logging and any code that reads the forward context. All segment replays within a step share the same context block.

## Expected Performance

| Scenario | GPU-CPU syncs/step | cudaGraphLaunch/step | CPU overhead/step |
|----------|-------------------|---------------------|-------------------|
| Eager + no cache-dit | 0 | 0 (per-kernel launch) | ~12ms |
| Whole graph (no cache-dit) | 0 | 1 | ~5us |
| Eager + cache-dit | 1 | 0 (per-kernel launch) | ~12ms |
| Step-level graphs + cache-dit | 1 | 3-4 | ~50-100us |

**Note on step-level overhead estimate**: The ~50-100us includes 3-4 × `cudaGraphLaunch()` (~3-5us each = ~15us), 1 × `.item()` GPU-CPU sync (~30-500us depending on GPU queue depth, typically ~50us when preceding graph is short), and 1-2 × `tensor.copy_()` (~5us each). The `.item()` sync dominates. This is still a **100-200× improvement** over eager mode's ~12ms.

Step-level graphs eliminate nearly all CPU kernel launch overhead while preserving cache-dit's step-level optimization decisions.

## Memory Estimate

### Graph Memory (shared pool)

| Resolution | Latent Shape | Approx. Graph Memory |
|-----------|-------------|---------------------|
| 256x256 | (1,16,1,32,32) | ~50MB |
| 512x512 | (1,16,1,64,64) | ~200MB |
| 1024x1024 | (1,16,1,128,128) | ~800MB |

With shared pool, total graph memory for all three resolutions ≈ largest (~800MB).

### Buffer Memory Per Resolution

ZImage: `hidden_size = 3840`, `dtype = BF16 (2 bytes)`.

The `inter_buffer` shape is `(1, seq_len + cap_seq_len, 3840)` where `cap_seq_len` is the caption token count (variable, typically ~256 tokens).

| Resolution | unified shape (approx.) | inter_buffer | cached_outputs | fn_output_snapshot | adaln_input | output_buffer | Total buffers |
|-----------|------------------------|-------------|---------------|-------------------|------------|--------------|--------------|
| 256x256 | (1, 512, 3840) | 3.9 MB | 3.9 MB | 3.9 MB | ~15 KB | ~0.1 MB | ~12 MB |
| 512x512 | (1, 1280, 3840) | 9.8 MB | 9.8 MB | 9.8 MB | ~15 KB | ~0.5 MB | ~30 MB |
| 1024x1024 | (1, 4352, 3840) | 33.4 MB | 33.4 MB | 33.4 MB | ~15 KB | ~2 MB | ~102 MB |

Buffer memory is small relative to graph memory and model weights. All three resolutions' buffers together ≈ **~144 MB**.

### Total Memory Overhead

```
Graph memory (shared pool): ~800 MB (dominated by 1024x1024)
Buffer memory (3 resolutions): ~144 MB
Total additional memory: ~950 MB
```

This is on top of model weights (~7.6B params × 2 bytes BF16 ≈ 15.2 GB, or less with FP8).

## Future Extensions

1. **Per-block level (PrunedBlocks)**: If step-level proves insufficient, evaluate per-block CUDA Graph approaches (conditional graph nodes, or piecewise per-block capture). Requires separate design.
2. **Other models**: Generalize segmented forward interface beyond ZImage. Requires model-specific `forward_pre_block` / `forward_post_block` implementations.
3. **CFG support**: Capture positive and negative forward passes as separate graphs or within a single graph.
4. **torch.compile integration**: Evaluate `torch.compile(mode="reduce-overhead")` as alternative or complement.
5. **SCM cache step optimization**: Split `graph_pre` into `graph_pre_timestep` (only timestep embed → adaln_input) and `graph_pre_full` to avoid wasted refiner computation on SCM cache steps.
6. **In-place inter_buffer writes (Phase 2)**: Eliminate per-segment `inter_buffer.copy_(out)` by having `forward_*_blocks()` write directly into pre-allocated `inter_buffer`. Saves ~25-32us per step at 1024×1024.
