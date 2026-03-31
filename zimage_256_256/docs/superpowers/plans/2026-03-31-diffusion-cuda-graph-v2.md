# Diffusion CUDA Graph v2 — Step-Level Piecewise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add step-level piecewise CUDA Graph capture to the ZImage-Turbo diffusion pipeline, compatible with cache-dit's SCM and DBCache step-level decisions.

**Architecture:** Split `dit.forward()` into 5 segments (pre/fn/middle/bn/post), capture each as an independent CUDA Graph, insert an eager decision point between fn and middle segments for DBCache's `can_cache()`. Falls back to the existing whole-graph `DiffusionCudaGraphRunner` when cache-dit is disabled.

**Tech Stack:** PyTorch CUDA Graphs, SGLang multimodal_gen, cache-dit library, ZImage-Turbo DiT model.

**Key constraint:** This is a GPU project — NO GPU execution on the local machine. All changes are file edits + static analysis. Testing happens on the remote GPU cluster.

**Spec:** `zimage_256_256/docs/superpowers/specs/2026-03-31-diffusion-cuda-graph-v2-design.md` (Revision 4, 3 rounds of review).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `python/sglang/multimodal_gen/envs.py` | Modify | Add `SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES` env var |
| `python/sglang/multimodal_gen/runtime/models/dits/zimage.py` | Modify | Add `PreBlockOutput` dataclass + 6 segmented forward methods |
| `python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py` | Modify | Add `StepLevelCudaGraphRunner` class; update existing `DiffusionCudaGraphRunner` to accept external `pool` |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` | Modify | Runner selection logic, shared pool, warmup, step-level replay loop integration |

---

## Task 1: Add Environment Variable

**Files:**
- Modify: `python/sglang/multimodal_gen/envs.py`

- [ ] **Step 1: Add TYPE_CHECKING entry**

In `envs.py`, find the line `SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D: bool = False` (currently line 58) and add the new env var **before** it, after the cache-dit secondary vars block:

```python
    # CUDA Graph warmup resolutions for diffusion
    SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES: str = "(256,256),(512,512),(1024,1024)"
```

- [ ] **Step 2: Add environment_variables entry**

Find the `"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D"` entry in the `environment_variables` dict (around line 248) and add immediately **before** it:

```python
    # CUDA Graph warmup resolutions
    "SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES": _lazy_str(
        "SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES",
        "(256,256),(512,512),(1024,1024)",
    ),
```

- [ ] **Step 3: Verify with static check**

Run: `python3 -c "from sglang.multimodal_gen import envs; print(type(envs))"`
Expected: `<class 'module'>` (no import errors)

- [ ] **Step 4: Commit**

```bash
git add python/sglang/multimodal_gen/envs.py
git commit -m "feat(diffusion): add SGLANG_DIFFUSION_CUDA_GRAPH_WARMUP_SIZES env var

Configurable list of (width, height) tuples for pre-capturing CUDA
Graphs at server startup. Defaults to 256x256, 512x512, 1024x1024."
```

---

## Task 2: Add PreBlockOutput Dataclass and Segmented Forward Methods to ZImage

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/models/dits/zimage.py`

This is the largest task. It adds 6 new methods + 1 dataclass to `ZImageTransformer2DModel` by extracting logic from the existing `forward()` method. The existing `forward()` is NOT modified — it continues to work unchanged for eager and whole-graph fallback.

- [ ] **Step 1: Add `dataclasses` import and `PreBlockOutput` dataclass**

At the top of `zimage.py`, add `from dataclasses import dataclass` to imports (line 1 area). Then, **before** the `class ZImageTransformer2DModel` definition (before line 615), add:

```python
from dataclasses import dataclass


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
        num_replicated_suffix: Number of replicated suffix tokens.
    """
    unified: torch.Tensor
    unified_freqs_cis: Tuple[torch.Tensor, torch.Tensor]
    adaln_input: torch.Tensor
    x_size: list
    x_local_seq_len: int
    use_full_unified_sequence: bool
    num_replicated_suffix: int
```

- [ ] **Step 2: Add `_get_raw_block()` method**

Add to `ZImageTransformer2DModel` class, after the existing `forward()` method (after line 995):

```python
    def _get_raw_block(self, block_idx: int):
        """Get the underlying ZImageTransformerBlock, bypassing CachedBlock wrapper.

        When cache-dit is enabled, self.layers[i] may be a CachedBlock wrapper.
        This method returns the raw block for direct execution in CUDA Graph
        capture. Cache-dit state synchronization is handled externally by
        StepLevelCudaGraphRunner.
        """
        layer = self.layers[block_idx]
        if hasattr(layer, "transformer_blocks"):
            return layer.transformer_blocks[0]
        return layer
```

- [ ] **Step 3: Add `forward_pre_block()` method**

This method extracts lines 864-947 of the existing `forward()`. Add after `_get_raw_block()`:

```python
    def forward_pre_block(
        self,
        hidden_states: List[torch.Tensor],
        encoder_hidden_states: List[torch.Tensor],
        timestep,
        guidance=0,
        patch_size=2,
        f_patch_size=1,
        freqs_cis=None,
    ) -> PreBlockOutput:
        """Pre-block segment: timestep embed -> patchify -> refiners -> concat.

        Extracts logic from forward() lines 864-947. The original forward()
        is preserved unchanged for eager/whole-graph fallback.
        """
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        x = hidden_states
        cap_feats = encoder_hidden_states
        timestep = 1000.0 - timestep
        t = timestep

        # Timestep embedding — use explicit dtype instead of type_as(list)
        t = self.t_embedder(t)
        adaln_input = t.to(dtype=self.t_embedder.mlp[0].weight.dtype)

        # Patchify and embed
        (
            x,
            cap_feats,
            x_size,
            x_valid_lens,
            cap_valid_lens,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # X embed linear
        x = torch.cat(x, dim=0)
        x, _ = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        # Unconditional padding for CUDA Graph robustness (M5)
        x[x_valid_lens[0]:] = self.x_pad_token.to(dtype=x.dtype)
        x_freqs_cis = freqs_cis[1]

        # Noise refiner
        x = x.unsqueeze(0)
        for layer in self.noise_refiner:
            x = layer(x, x_freqs_cis, adaln_input)

        # Cap embed
        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats, _ = self.cap_embedder(cap_feats)
        # Unconditional padding for CUDA Graph robustness (M5)
        cap_feats[cap_valid_lens[0]:] = self.cap_pad_token.to(dtype=cap_feats.dtype)
        cap_freqs_cis = freqs_cis[0]

        # Context refiner
        cap_feats = cap_feats.unsqueeze(0)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs_cis)

        # Unified concat and setup
        cap_seq_len = cap_feats.shape[1]
        use_full_unified_sequence = (
            get_sp_world_size() > 1 and get_ring_parallel_world_size() > 1
        )
        x_local_seq_len = x.shape[1]
        if use_full_unified_sequence:
            x = sequence_model_parallel_all_gather(x.contiguous(), dim=1)
            x_freqs_cis = (
                sequence_model_parallel_all_gather(x_freqs_cis[0].contiguous(), dim=0),
                sequence_model_parallel_all_gather(x_freqs_cis[1].contiguous(), dim=0),
            )
        unified = torch.cat([x, cap_feats], dim=1)
        unified_freqs_cis = (
            torch.cat([x_freqs_cis[0], cap_freqs_cis[0]], dim=0),
            torch.cat([x_freqs_cis[1], cap_freqs_cis[1]], dim=0),
        )
        num_replicated_suffix = cap_seq_len if not use_full_unified_sequence else 0

        return PreBlockOutput(
            unified=unified,
            unified_freqs_cis=unified_freqs_cis,
            adaln_input=adaln_input,
            x_size=x_size,
            x_local_seq_len=x_local_seq_len,
            use_full_unified_sequence=use_full_unified_sequence,
            num_replicated_suffix=num_replicated_suffix,
        )
```

- [ ] **Step 4: Add `forward_fn_blocks()` method**

```python
    def forward_fn_blocks(
        self,
        unified: torch.Tensor,
        unified_freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        adaln_input: torch.Tensor,
        num_replicated_suffix: int,
        fn_count: int,
        use_full_unified_sequence: bool = False,
    ) -> torch.Tensor:
        """Execute first fn_count main transformer blocks (always full compute).

        Bypasses CachedBlock wrappers and calls raw ZImageTransformerBlock directly.
        """
        for i in range(fn_count):
            raw_block = self._get_raw_block(i)
            raw_block.attention.attn.skip_sequence_parallel = use_full_unified_sequence
            unified = raw_block(
                unified,
                unified_freqs_cis,
                adaln_input,
                num_replicated_suffix=num_replicated_suffix,
            )
        return unified
```

- [ ] **Step 5: Add `forward_middle_blocks()` method**

```python
    def forward_middle_blocks(
        self,
        unified: torch.Tensor,
        unified_freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        adaln_input: torch.Tensor,
        num_replicated_suffix: int,
        fn_count: int,
        bn_count: int,
        use_full_unified_sequence: bool = False,
    ) -> torch.Tensor:
        """Execute middle blocks (Fn..N-Bn-1) in full compute mode."""
        end = len(self.layers) - bn_count
        for i in range(fn_count, end):
            raw_block = self._get_raw_block(i)
            raw_block.attention.attn.skip_sequence_parallel = use_full_unified_sequence
            unified = raw_block(
                unified,
                unified_freqs_cis,
                adaln_input,
                num_replicated_suffix=num_replicated_suffix,
            )
        return unified
```

- [ ] **Step 6: Add `forward_bn_blocks()` method**

```python
    def forward_bn_blocks(
        self,
        unified: torch.Tensor,
        unified_freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        adaln_input: torch.Tensor,
        num_replicated_suffix: int,
        bn_count: int,
        use_full_unified_sequence: bool = False,
    ) -> torch.Tensor:
        """Execute last bn_count blocks (always full compute).

        Not called when bn_count=0. Caller skips this entirely when Bn=0.
        """
        if bn_count == 0:
            return unified
        start = len(self.layers) - bn_count
        for i in range(start, len(self.layers)):
            raw_block = self._get_raw_block(i)
            raw_block.attention.attn.skip_sequence_parallel = use_full_unified_sequence
            unified = raw_block(
                unified,
                unified_freqs_cis,
                adaln_input,
                num_replicated_suffix=num_replicated_suffix,
            )
        return unified
```

- [ ] **Step 7: Add `forward_post_block()` method**

```python
    def forward_post_block(
        self,
        unified: torch.Tensor,
        adaln_input: torch.Tensor,
        patch_size: int,
        f_patch_size: int,
        x_size: list,
        x_local_seq_len: int,
        use_full_unified_sequence: bool,
    ) -> torch.Tensor:
        """Post-block segment: final_layer -> SP restore -> unpatchify -> output.

        Contains Python-level list/reshape ops that are baked in during capture.
        Safe for CUDA Graph because shapes are fixed per resolution and 2x warmup
        stabilizes the allocator.

        Returns: noise_pred tensor (single torch.Tensor).
        """
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified, adaln_input
        )

        if use_full_unified_sequence:
            sp_rank = get_sp_parallel_rank()
            start = sp_rank * x_local_seq_len
            end = start + x_local_seq_len
            unified = unified[:, start:end]
        x = list(unified.unbind(dim=0))
        x = self.unpatchify(x, x_size, patch_size, f_patch_size)

        return -x[0]
```

- [ ] **Step 8: Verify import and syntax**

Run: `python3 -c "import ast; ast.parse(open('python/sglang/multimodal_gen/runtime/models/dits/zimage.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/models/dits/zimage.py
git commit -m "feat(diffusion): add segmented forward methods to ZImage DiT

Add PreBlockOutput dataclass and 6 methods to ZImageTransformer2DModel:
- forward_pre_block: timestep embed -> patchify -> refiners -> concat
- forward_fn_blocks: first Fn blocks (always full compute)
- forward_middle_blocks: middle blocks (Fn..N-Bn-1)
- forward_bn_blocks: last Bn blocks (always full compute)
- forward_post_block: final_layer -> unpatchify -> output
- _get_raw_block: bypass CachedBlock wrapper

These extract logic from the existing forward() without modifying it.
Used by StepLevelCudaGraphRunner for piecewise CUDA Graph capture."
```

---

## Task 3: Update DiffusionCudaGraphRunner to Accept External Pool

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py`

- [ ] **Step 1: Add optional `pool` parameter to `__init__`**

Change `DiffusionCudaGraphRunner.__init__` (line 92) from:

```python
    def __init__(self, device: torch.device):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        # Shared memory pool — avoids fragmentation across captures.
        self.pool = torch.cuda.graph_pool_handle()
```

to:

```python
    def __init__(self, device: torch.device, pool=None):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        # Shared memory pool — avoids fragmentation across captures.
        # If pool is provided externally, use it (shared across runners).
        # Otherwise create a private pool (backward compatible).
        self.pool = pool if pool is not None else torch.cuda.graph_pool_handle()
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py
git commit -m "refactor(diffusion): DiffusionCudaGraphRunner accepts external pool

Allow passing a shared graph_pool_handle so all runners (both whole-graph
and step-level) share the same memory pool. Falls back to creating a
private pool if none provided (backward compatible)."
```

---

## Task 4: Implement StepLevelCudaGraphRunner

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py`

- [ ] **Step 1: Add imports at top of file**

After the existing imports (line 23 area), add:

```python
import ast
from dataclasses import dataclass

from sglang.multimodal_gen import envs
```

- [ ] **Step 2: Add `StepLevelCudaGraphRunner` class**

Add after the `DiffusionCudaGraphRunner` class (after line 232):

```python
class StepLevelCudaGraphRunner:
    """Step-level piecewise CUDA Graph runner for cache-dit compatible inference.

    Captures dit.forward() as multiple segments (pre/fn/middle/bn/post) with an
    eager decision point between fn and middle segments, enabling DBCache
    step-level decisions at replay time.

    When cache-dit is disabled, use DiffusionCudaGraphRunner instead (whole graph).

    Preconditions (enforced by caller before capture):
    - Attention backend must NOT use current_timestep for kernel path selection
    - boundary_timestep must be None (no mid-request model switching)
    - cache-dit must be enabled (otherwise use DiffusionCudaGraphRunner)

    Usage:
        # First request: capture
        runner = StepLevelCudaGraphRunner(device, 30, fn=1, bn=0, pool=pool)
        runner.capture(model, timestep, latents, static_kwargs)

        # Same request, subsequent steps:
        runner.replay_step(timestep, latents, step_is_all_cached=False)

        # Next request:
        runner.update_static_kwargs(new_static_kwargs)
        runner.replay_step(timestep, latents, step_is_all_cached=scm_mask[i]==0)
    """

    def __init__(
        self,
        device: torch.device,
        num_blocks: int,
        fn_blocks: int,
        bn_blocks: int,
        pool,
    ):
        self.device = device
        self.num_blocks = num_blocks
        self.fn_blocks = fn_blocks
        self.bn_blocks = bn_blocks
        self.pool = pool

        # Graphs — None until captured
        self.graph_pre: Optional[torch.cuda.CUDAGraph] = None
        self.graph_fn: Optional[torch.cuda.CUDAGraph] = None
        self.graph_middle_full: Optional[torch.cuda.CUDAGraph] = None
        self.graph_bn: Optional[torch.cuda.CUDAGraph] = None  # None when bn_blocks==0
        self.graph_post: Optional[torch.cuda.CUDAGraph] = None

        # Fixed-address buffers (allocated at capture time)
        self.input_buffers: dict[str, object] = {}
        self.static_buffers: dict[str, object] = {}
        self.inter_buffer: Optional[torch.Tensor] = None
        self.adaln_input_buffer: Optional[torch.Tensor] = None
        self.unified_freqs_cis_buffer: Optional[tuple] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.cached_outputs: Optional[torch.Tensor] = None
        self._fn_output_snapshot: Optional[torch.Tensor] = None

        # Metadata from pre_block (fixed per resolution)
        self._x_size: Optional[list] = None
        self._x_local_seq_len: Optional[int] = None
        self._use_full_unified_sequence: Optional[bool] = None
        self._num_replicated_suffix: Optional[int] = None
        self._patch_size: Optional[int] = None
        self._f_patch_size: Optional[int] = None

        # Reference to the model (set at capture time)
        self._transformer = None

        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def _capture_one(self, fn, warmup_runs=_WARMUP_RUNS):
        """Warmup + capture a single segment into a CUDA Graph."""
        for _ in range(warmup_runs):
            fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            fn()
        return graph

    def capture(
        self,
        dit_model,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        static_kwargs: dict,
    ) -> torch.Tensor:
        """Capture all segments. Called once per resolution.

        Must be called within a set_forward_context() block to ensure
        warmup runs and the capture run share the same forward context.

        Phase 1: One eager forward to determine shapes and allocate buffers.
        Phase 2: Capture each segment sequentially (pre -> fn -> middle -> [bn] -> post).
        """
        self._transformer = dit_model
        self._patch_size = static_kwargs.get("patch_size", 2)
        self._f_patch_size = static_kwargs.get("f_patch_size", 1)

        # Create persistent buffers for inputs
        timestep_buffer = timestep.clone()
        latent_buffer = _deep_clone_structure(latents)
        self.input_buffers = {
            "timestep": timestep_buffer,
            "latent": latent_buffer,
        }

        # Create persistent buffers for static kwargs
        self.static_buffers = {}
        graph_static_kwargs = {}
        for key, val in static_kwargs.items():
            buf = _deep_clone_structure(val)
            self.static_buffers[key] = buf
            graph_static_kwargs[key] = buf

        # --- Phase 1: Eager forward to determine shapes ---
        logger.info("StepLevel CUDA Graph: Phase 1 — shape discovery")
        with torch.no_grad():
            pre_out = dit_model.forward_pre_block(
                hidden_states=latent_buffer,
                encoder_hidden_states=graph_static_kwargs.get("encoder_hidden_states"),
                timestep=timestep_buffer,
                guidance=graph_static_kwargs.get("guidance", 0),
                patch_size=self._patch_size,
                f_patch_size=self._f_patch_size,
                freqs_cis=graph_static_kwargs.get("freqs_cis"),
            )

        # Store metadata (fixed per resolution)
        self._x_size = pre_out.x_size
        self._x_local_seq_len = pre_out.x_local_seq_len
        self._use_full_unified_sequence = pre_out.use_full_unified_sequence
        self._num_replicated_suffix = pre_out.num_replicated_suffix

        # Allocate fixed-address buffers
        self.inter_buffer = torch.empty_like(pre_out.unified)
        self.adaln_input_buffer = torch.empty_like(pre_out.adaln_input)
        self.unified_freqs_cis_buffer = (
            pre_out.unified_freqs_cis[0].clone(),
            pre_out.unified_freqs_cis[1].clone(),
        )
        self.cached_outputs = torch.empty_like(pre_out.unified)
        self._fn_output_snapshot = torch.empty_like(pre_out.unified)

        # --- Phase 2: Capture each segment ---
        logger.info("StepLevel CUDA Graph: Phase 2 — capturing segments")

        # Capture pre_block
        def pre_fn():
            result = dit_model.forward_pre_block(
                hidden_states=self.input_buffers["latent"],
                encoder_hidden_states=self.static_buffers.get("encoder_hidden_states"),
                timestep=self.input_buffers["timestep"],
                guidance=self.static_buffers.get("guidance", 0),
                patch_size=self._patch_size,
                f_patch_size=self._f_patch_size,
                freqs_cis=self.static_buffers.get("freqs_cis"),
            )
            self.inter_buffer.copy_(result.unified)
            self.adaln_input_buffer.copy_(result.adaln_input)

        self.graph_pre = self._capture_one(pre_fn)
        logger.info("  Captured graph_pre")

        # Capture fn_blocks
        nrs = self._num_replicated_suffix
        ufs = self._use_full_unified_sequence

        def fn_fn():
            out = dit_model.forward_fn_blocks(
                self.inter_buffer,
                self.unified_freqs_cis_buffer,
                self.adaln_input_buffer,
                num_replicated_suffix=nrs,
                fn_count=self.fn_blocks,
                use_full_unified_sequence=ufs,
            )
            self.inter_buffer.copy_(out)

        self.graph_fn = self._capture_one(fn_fn)
        logger.info("  Captured graph_fn (Fn=%d blocks)", self.fn_blocks)

        # Capture middle_blocks
        def middle_fn():
            out = dit_model.forward_middle_blocks(
                self.inter_buffer,
                self.unified_freqs_cis_buffer,
                self.adaln_input_buffer,
                num_replicated_suffix=nrs,
                fn_count=self.fn_blocks,
                bn_count=self.bn_blocks,
                use_full_unified_sequence=ufs,
            )
            self.inter_buffer.copy_(out)

        self.graph_middle_full = self._capture_one(middle_fn)
        logger.info(
            "  Captured graph_middle_full (blocks %d..%d)",
            self.fn_blocks,
            self.num_blocks - self.bn_blocks - 1,
        )

        # Capture bn_blocks (only if Bn > 0)
        if self.bn_blocks > 0:
            def bn_fn():
                out = dit_model.forward_bn_blocks(
                    self.inter_buffer,
                    self.unified_freqs_cis_buffer,
                    self.adaln_input_buffer,
                    num_replicated_suffix=nrs,
                    bn_count=self.bn_blocks,
                    use_full_unified_sequence=ufs,
                )
                self.inter_buffer.copy_(out)

            self.graph_bn = self._capture_one(bn_fn)
            logger.info("  Captured graph_bn (Bn=%d blocks)", self.bn_blocks)

        # Capture post_block
        def post_fn():
            out = dit_model.forward_post_block(
                self.inter_buffer,
                self.adaln_input_buffer,
                patch_size=self._patch_size,
                f_patch_size=self._f_patch_size,
                x_size=self._x_size,
                x_local_seq_len=self._x_local_seq_len,
                use_full_unified_sequence=ufs,
            )
            self.output_buffer = out

        self.graph_post = self._capture_one(post_fn)
        logger.info("  Captured graph_post")

        self._captured = True
        n_graphs = 4 + (1 if self.bn_blocks > 0 else 0)
        logger.info(
            "StepLevel CUDA Graph: capture complete (%d graphs, Fn=%d, Bn=%d)",
            n_graphs,
            self.fn_blocks,
            self.bn_blocks,
        )
        return self.output_buffer

    def update_static_kwargs(self, static_kwargs: dict):
        """Copy new per-request data into persistent static buffers.

        Called once at the start of each new request.
        """
        assert self._captured, "Must call capture() before update_static_kwargs()"
        for key, new_val in static_kwargs.items():
            if key in self.static_buffers:
                _deep_copy_to_buffer(new_val, self.static_buffers[key])

    def replay_step(
        self,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        step_is_all_cached: bool,
    ) -> torch.Tensor:
        """Replay one denoising step with step-level cache-dit decision.

        Args:
            timestep: Current timestep tensor.
            latents: Current latent tensor.
            step_is_all_cached: True if SCM says this step is all-cache
                (skip Fn/middle/Bn, use cached_outputs directly).

        Returns:
            noise_pred from output_buffer.
        """
        assert self._captured, "Must call capture() before replay_step()"

        # Update per-step inputs
        _deep_copy_to_buffer(latents, self.input_buffers["latent"])
        self.input_buffers["timestep"].copy_(timestep)

        # Pre-block (every step — produces adaln_input for post_block)
        self.graph_pre.replay()

        if step_is_all_cached:
            # === SCM all-cache fast path ===
            self.inter_buffer.copy_(self.cached_outputs)
        else:
            # === DBCache decision path ===
            self.graph_fn.replay()

            # Eager: query DBCache decision
            step_can_cache = self._query_dbcache_decision()

            if step_can_cache:
                self.inter_buffer.copy_(self.cached_outputs)
            else:
                self.graph_middle_full.replay()
                self.cached_outputs.copy_(self.inter_buffer)
                self._update_fn_wrapper_cached_output()

            # Bn blocks (only if Bn > 0)
            if self.graph_bn is not None:
                self.graph_bn.replay()

        # Post-block (every step)
        self.graph_post.replay()

        return self.output_buffer

    def _query_dbcache_decision(self) -> bool:
        """Call cache-dit's can_cache() in eager mode after Fn blocks replay.

        Saves Fn output snapshot for _update_fn_wrapper_cached_output().
        Returns True if remaining blocks can use cache.
        """
        # Save Fn output for later state update
        self._fn_output_snapshot.copy_(self.inter_buffer)

        ctx_mgr = getattr(self._transformer, "_context_manager", None)
        if ctx_mgr is None:
            # No cache-dit context manager — always compute
            return False

        return ctx_mgr.can_cache(self.inter_buffer)

    def _update_fn_wrapper_cached_output(self):
        """Update cache-dit's internal state after a full-compute step.

        The exact attribute name on CachedContextManager must be confirmed
        by the eager-mode prototype. This is a best-effort implementation
        that will be refined during integration testing.
        """
        ctx_mgr = getattr(self._transformer, "_context_manager", None)
        if ctx_mgr is None:
            return

        # Update similarity baseline with current Fn output.
        # Try known attribute names — exact API confirmed in prototype.
        if hasattr(ctx_mgr, "_cached_residual"):
            ctx_mgr._cached_residual.copy_(self._fn_output_snapshot)
        elif hasattr(ctx_mgr, "cached_output"):
            ctx_mgr.cached_output.copy_(self._fn_output_snapshot)

    def reset(self):
        """Release all captured graphs and buffers."""
        for attr in ("graph_pre", "graph_fn", "graph_middle_full", "graph_bn", "graph_post"):
            graph = getattr(self, attr, None)
            if graph is not None:
                del graph
                setattr(self, attr, None)
        self.input_buffers.clear()
        self.static_buffers.clear()
        self.inter_buffer = None
        self.adaln_input_buffer = None
        self.unified_freqs_cis_buffer = None
        self.output_buffer = None
        self.cached_outputs = None
        self._fn_output_snapshot = None
        self._transformer = None
        self._captured = False
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py
git commit -m "feat(diffusion): add StepLevelCudaGraphRunner for cache-dit compatibility

New runner that captures dit.forward() as 4-5 piecewise CUDA Graphs
(pre/fn/middle/bn/post) with an eager decision point between fn and
middle segments. Enables DBCache step-level can_cache() decisions at
replay time while keeping GPU kernels in graphs.

Supports:
- SCM all-cache fast path (skip fn/middle/bn, use cached_outputs)
- DBCache compute/cache decision per step
- Shared pool across all runners
- Per-request static kwargs update"
```

---

## Task 5: Integrate into Denoising Loop

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`

This is the integration task that wires everything together. The changes are:
1. Shared pool management in `__init__`
2. Runner selection logic (StepLevel vs whole-graph)
3. SCM mask pre-computation
4. Step-level replay in the denoising loop

- [ ] **Step 1: Add imports**

At the top of `denoising.py`, after the existing `DiffusionCudaGraphRunner` import (line 62), add:

```python
from sglang.multimodal_gen.runtime.managers.diffusion_cuda_graph_runner import (
    DiffusionCudaGraphRunner,
    StepLevelCudaGraphRunner,
)
```

And remove the existing single import of `DiffusionCudaGraphRunner` (line 61-63) since it's now covered by the new multi-import.

Also add at the top:

```python
import ast
```

- [ ] **Step 2: Add shared pool to `__init__`**

In `DenoisingStage.__init__()`, after the existing `self._cuda_graph_runners` dict (line 138), add:

```python
        # Shared CUDA Graph memory pool across all runners (both types).
        # Created lazily on first use to avoid import-time CUDA init.
        self._graph_pool = None

        # Step-level runners: keyed by latent shape, used when cache-dit enabled.
        self._step_level_runners: dict[tuple, "StepLevelCudaGraphRunner"] = {}
```

- [ ] **Step 3: Add pool accessor method**

Add a method to `DenoisingStage`:

```python
    def _get_graph_pool(self):
        """Get or create the shared CUDA Graph memory pool."""
        if self._graph_pool is None:
            import torch
            self._graph_pool = torch.cuda.graph_pool_handle()
        return self._graph_pool
```

- [ ] **Step 4: Modify runner selection in `forward()`**

In the `forward()` method, replace the CUDA Graph setup block (lines 1002-1063) with the new logic. Find the section starting with `# CUDA Graph setup for denoising` and replace through the `logger.info` call:

```python
        # CUDA Graph setup for denoising
        cuda_graph_enabled = server_args.enable_diffusion_cuda_graph
        graph_runner = None  # DiffusionCudaGraphRunner (whole graph, no cache-dit)
        step_level_runner = None  # StepLevelCudaGraphRunner (cache-dit compatible)
        scm_mask = None

        if cuda_graph_enabled:
            # Runtime safety checks
            _TIMESTEP_DEPENDENT_BACKENDS = {
                AttentionBackendEnum.SLIDING_TILE_ATTN,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN,
                AttentionBackendEnum.VMOBA_ATTN,
            }
            assert self.attn_backend.get_enum() not in _TIMESTEP_DEPENDENT_BACKENDS, (
                f"Diffusion CUDA Graph is incompatible with attention backend "
                f"'{self.attn_backend.get_enum().name}'. These backends use "
                f"current_timestep for kernel path selection, which cannot "
                f"change after graph capture. Use FlashAttention instead."
            )
            assert boundary_timestep is None, (
                f"Diffusion CUDA Graph does not support mid-request model "
                f"switching (boundary_timestep={boundary_timestep}). All "
                f"denoising steps must use the same transformer."
            )

            static_kwargs = self.prepare_extra_func_kwargs(
                getattr(self.transformer, "forward", self.transformer),
                {
                    "encoder_hidden_states": pos_cond_kwargs.get(
                        "encoder_hidden_states"
                    ),
                    "guidance": guidance,
                    "freqs_cis": pos_cond_kwargs.get("freqs_cis"),
                },
            )

            latent_shape = tuple(latents.shape)
            pool = self._get_graph_pool()

            if self._cache_dit_enabled:
                # === Step-level runner (cache-dit compatible) ===
                step_level_runner = self._step_level_runners.get(latent_shape)

                if step_level_runner is not None:
                    step_level_runner.update_static_kwargs(static_kwargs)
                    logger.info(
                        "StepLevel CUDA Graph: reusing cached runner for shape %s",
                        latent_shape,
                    )
                else:
                    step_level_runner = StepLevelCudaGraphRunner(
                        device=get_local_torch_device(),
                        num_blocks=len(self.transformer.layers),
                        fn_blocks=envs.SGLANG_CACHE_DIT_FN,
                        bn_blocks=envs.SGLANG_CACHE_DIT_BN,
                        pool=pool,
                    )
                    logger.info(
                        "StepLevel CUDA Graph: will capture for shape %s "
                        "(Fn=%d, Bn=%d)",
                        latent_shape,
                        envs.SGLANG_CACHE_DIT_FN,
                        envs.SGLANG_CACHE_DIT_BN,
                    )

                # Pre-compute SCM mask
                scm_preset = envs.SGLANG_CACHE_DIT_SCM_PRESET
                if scm_preset != "none":
                    scm_mask = get_scm_mask(scm_preset, num_inference_steps)

            else:
                # === Whole-graph runner (no cache-dit) ===
                graph_runner = self._cuda_graph_runners.get(latent_shape)

                if graph_runner is not None:
                    graph_runner.update_static_kwargs(static_kwargs)
                    logger.info(
                        "Diffusion CUDA Graph: reusing cached graph for shape %s",
                        latent_shape,
                    )
                else:
                    graph_runner = DiffusionCudaGraphRunner(
                        device=get_local_torch_device(),
                        pool=pool,
                    )
                    logger.info(
                        "Diffusion CUDA Graph: will capture for shape %s",
                        latent_shape,
                    )
```

- [ ] **Step 5: Modify the denoising loop body**

Replace the inner `# Predict noise residual` section of the denoising loop (lines 1124-1179) with the three-way branch:

```python
                        # Predict noise residual
                        if step_level_runner is not None:
                            # === Step-level CUDA Graph (cache-dit compatible) ===
                            with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ):
                                if not step_level_runner.captured:
                                    # First request for this shape: capture
                                    noise_pred = step_level_runner.capture(
                                        dit_model=self.transformer,
                                        timestep=timestep,
                                        latents=latent_model_input,
                                        static_kwargs=static_kwargs,
                                    )
                                    self._step_level_runners[latent_shape] = step_level_runner
                                else:
                                    # Replay with SCM/DBCache decision
                                    step_is_all_cached = (
                                        scm_mask is not None and scm_mask[i] == 0
                                    )
                                    noise_pred = step_level_runner.replay_step(
                                        timestep=timestep,
                                        latents=latent_model_input,
                                        step_is_all_cached=step_is_all_cached,
                                    )

                        elif (
                            cuda_graph_enabled
                            and graph_runner is not None
                            and not graph_runner.captured
                        ):
                            # === Whole-graph capture (no cache-dit) ===
                            with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ):
                                noise_pred = graph_runner.capture(
                                    dit_forward_fn=current_model,
                                    timestep=timestep,
                                    latents=latent_model_input,
                                    static_kwargs=static_kwargs,
                                )
                            self._cuda_graph_runners[latent_shape] = graph_runner

                        elif (
                            cuda_graph_enabled
                            and graph_runner is not None
                            and graph_runner.captured
                        ):
                            # === Whole-graph replay (no cache-dit) ===
                            with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ):
                                noise_pred = graph_runner.replay(
                                    timestep, latent_model_input
                                )
                        else:
                            # === Eager fallback ===
                            attn_metadata = self._build_attn_metadata(
                                i,
                                batch,
                                server_args,
                                timestep_value=t_int,
                                timesteps=timesteps_cpu,
                            )
                            noise_pred = self._predict_noise_with_cfg(
                                current_model=current_model,
                                latent_model_input=latent_model_input,
                                timestep=timestep,
                                batch=batch,
                                timestep_index=i,
                                attn_metadata=attn_metadata,
                                target_dtype=target_dtype,
                                current_guidance_scale=current_guidance_scale,
                                image_kwargs=image_kwargs,
                                pos_cond_kwargs=pos_cond_kwargs,
                                neg_cond_kwargs=neg_cond_kwargs,
                                server_args=server_args,
                                guidance=guidance,
                                latents=latents,
                            )
```

- [ ] **Step 6: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py
git commit -m "feat(diffusion): integrate StepLevelCudaGraphRunner into denoising loop

Wire up the step-level piecewise CUDA Graph runner:
- Shared pool management across all runners
- Automatic runner selection: StepLevel when cache-dit ON, whole-graph when OFF
- SCM mask pre-computation for fast all-cache path
- Three-way branch in denoising loop: step-level / whole-graph / eager
- Lazy capture on first request per resolution"
```

---

## Task 6: Final Static Verification

- [ ] **Step 1: Verify all modified files parse cleanly**

```bash
python3 -c "
import ast, sys
files = [
    'python/sglang/multimodal_gen/envs.py',
    'python/sglang/multimodal_gen/runtime/models/dits/zimage.py',
    'python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py',
    'python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py',
]
for f in files:
    try:
        ast.parse(open(f).read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'FAIL: {f}: {e}')
        sys.exit(1)
print('All files OK')
"
```

Expected: All files OK

- [ ] **Step 2: Verify no circular imports**

```bash
python3 -c "from sglang.multimodal_gen import envs; print('envs OK')"
```

Expected: `envs OK`

- [ ] **Step 3: Commit all if not already done**

```bash
git status
```

Verify all changes are committed. If any unstaged changes remain, add and commit.

---

## Testing Notes (for remote GPU cluster)

The following tests must be run on the GPU cluster, not locally:

1. **Eager baseline**: Run ZImage-Turbo inference without CUDA Graph (baseline correctness + timing)
2. **Whole-graph**: Run with `--enable-diffusion-cuda-graph` and cache-dit OFF — verify existing behavior preserved
3. **Step-level**: Run with `--enable-diffusion-cuda-graph` and `SGLANG_CACHE_DIT_ENABLED=1` — verify output matches eager
4. **Eager prototype**: Run segmented forward methods in eager mode, compare `can_cache()` decisions against CachedBlock wrapper baseline — validates cache-dit state consistency (spec requirement: "Pre-Implementation Validation")
5. **Multi-resolution**: Test warmup with default sizes + lazy capture for non-warmup resolution
6. **Performance**: Compare latency: eager vs whole-graph vs step-level (expect step-level ~50-100us overhead vs eager's ~12ms)
