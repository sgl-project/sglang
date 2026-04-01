# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph runner for diffusion DiT forward pass.

Captures a single DiT forward pass and replays it with updated inputs.
Analogous to srt's CudaGraphRunner but scoped to diffusion models.

Graph boundary:
    Captures:  dit.forward()  (model inference only)
    Excludes:  scheduler.step(), timestep expansion, CFG combine, profiling

Cross-request caching:
    The runner is cached on the DenoisingStage instance keyed by latent
    shape. On the first request, warmup + capture occurs. On subsequent
    requests with the same shape, only buffer copies + replay happen.
    Per-request inputs (encoder_hidden_states, freqs_cis, guidance,
    timestep, latents) are copied into persistent fixed-address buffers
    before each replay.
"""

from typing import Callable, Optional

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_WARMUP_RUNS = 2


def _deep_copy_to_buffer(src, dst):
    """Copy tensor data from src into dst, handling nested structures.

    Supports: Tensor, List[Tensor], Tuple[Tensor, ...], and nested
    combinations. src and dst must have identical structure and shapes.
    """
    if isinstance(src, torch.Tensor):
        dst.copy_(src)
    elif isinstance(src, (list, tuple)):
        for s, d in zip(src, dst):
            _deep_copy_to_buffer(s, d)
    # Non-tensor types (int, float, None) — no copy needed


def _deep_clone_structure(obj):
    """Create persistent buffer copies of a nested tensor structure.

    Returns a structure with the same nesting but each tensor replaced
    by a .clone() (new memory address, stable for CUDA Graph).
    Non-tensor leaves are returned as-is.
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, list):
        return [_deep_clone_structure(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_clone_structure(x) for x in obj)
    else:
        return obj


class DiffusionCudaGraphRunner:
    """CUDA Graph runner for diffusion DiT forward pass.

    Captures a single DiT forward pass and replays it with updated inputs.
    On replay, all GPU kernels execute from a single cudaGraphLaunch() call,
    eliminating CPU-side kernel launch latency between kernels.

    The runner is designed to be cached across requests. All inputs —
    including per-request ones like encoder_hidden_states — are held in
    persistent fixed-address buffers. New request data is copied into
    these buffers before replay.

    Preconditions (enforced by caller before capture):
    - Attention backend must NOT use current_timestep for kernel path selection
    - boundary_timestep must be None (no mid-request model switching)

    Usage:
        # First request: capture
        runner = DiffusionCudaGraphRunner(device)
        runner.capture(model, timestep, latents, static_kwargs)

        # Same request, subsequent steps:
        runner.replay(new_timestep, new_latents)

        # Next request (different prompt, same resolution):
        runner.update_static_kwargs(new_static_kwargs)
        runner.replay(timestep, latents)  # no re-capture needed
    """

    def __init__(self, device: torch.device, pool=None):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        # Shared memory pool — avoids fragmentation across captures.
        # If pool is provided externally, use it (shared across runners).
        # Otherwise create a private pool (backward compatible).
        self.pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self.input_buffers: dict[str, torch.Tensor] = {}
        # Persistent buffers for static kwargs (encoder_hidden_states, etc.)
        self.static_buffers: dict[str, object] = {}
        self.output_buffer: Optional[torch.Tensor] = None
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def capture(
        self,
        dit_forward_fn: Callable,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        static_kwargs: dict,
    ) -> torch.Tensor:
        """Capture one DiT forward pass into a CUDA Graph.

        Must be called within a set_forward_context() block.

        Creates persistent buffers for ALL inputs (timestep, latents,
        and every tensor in static_kwargs). The graph records these
        buffer addresses. Subsequent calls to replay() or
        update_static_kwargs() copy new data into these buffers.

        Args:
            dit_forward_fn: The model's forward callable.
            timestep: Initial timestep tensor (will be cloned into buffer).
            latents: Initial latent tensor (will be cloned into buffer).
            static_kwargs: Per-request model inputs (encoder_hidden_states,
                guidance, freqs_cis, etc.). All tensors are cloned into
                persistent buffers.

        Returns:
            Output tensor (noise_pred). Address is fixed for replay.
        """
        # Create persistent buffers for dynamic inputs
        timestep_buffer = timestep.clone()
        latent_buffer = latents.clone()
        self.input_buffers = {
            "timestep": timestep_buffer,
            "latent": latent_buffer,
        }

        # Create persistent buffers for static kwargs.
        # Each tensor in static_kwargs is cloned so we own the memory
        # address. CUDA Graph bakes in these addresses.
        self.static_buffers = {}
        graph_static_kwargs = {}
        for key, val in static_kwargs.items():
            buf = _deep_clone_structure(val)
            self.static_buffers[key] = buf
            graph_static_kwargs[key] = buf

        # Build the callable that the graph will capture.
        # Note: ZImage's eager path passes hidden_states as a plain
        # tensor, not a list, despite the List[Tensor] type hint.
        def run_fn():
            return dit_forward_fn(
                hidden_states=latent_buffer,
                timestep=timestep_buffer,
                **graph_static_kwargs,
            )

        # Warmup: 2 eager runs to stabilize PyTorch's caching allocator.
        # Run 1 may trigger new CUDA memory allocations.
        # Run 2 confirms steady state (no new allocations).
        logger.info(
            "CUDA Graph: warming up with %d eager runs before capture",
            _WARMUP_RUNS,
        )
        for _ in range(_WARMUP_RUNS):
            run_fn()
        torch.cuda.synchronize()

        # Capture
        logger.info("CUDA Graph: capturing dit.forward()")
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=self.pool):
            output = run_fn()

        self.output_buffer = output
        self._captured = True
        logger.info("CUDA Graph: capture complete")
        return output

    def update_static_kwargs(self, static_kwargs: dict):
        """Copy new per-request data into persistent static buffers.

        Call this once at the start of each new request (before the
        denoising loop) to update encoder_hidden_states, freqs_cis, etc.
        The buffer addresses stay the same — only data changes.

        Args:
            static_kwargs: New per-request static kwargs. Must have the
                same keys and tensor shapes as those used during capture.
        """
        assert self._captured, "Must call capture() before update_static_kwargs()"
        for key, new_val in static_kwargs.items():
            if key in self.static_buffers:
                _deep_copy_to_buffer(new_val, self.static_buffers[key])

    def replay(
        self,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Update per-step inputs and replay the captured graph.

        Args:
            timestep: New timestep value for this denoising step.
            latents: New latent input for this denoising step.

        Returns:
            noise_pred from the output buffer.
        """
        assert self._captured, "Must call capture() before replay()"

        self.input_buffers["timestep"].copy_(timestep)
        self.input_buffers["latent"].copy_(latents)

        self.graph.replay()

        return self.output_buffer

    def reset(self):
        """Release the captured graph and all buffers."""
        if self.graph is not None:
            del self.graph
            self.graph = None
        self.input_buffers.clear()
        self.static_buffers.clear()
        self.output_buffer = None
        self._captured = False


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
        self.graph_bn: Optional[torch.cuda.CUDAGraph] = None
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
        self._fn_input_snapshot: Optional[torch.Tensor] = None

        # Metadata from pre_block (fixed per resolution)
        self._x_size: Optional[list] = None
        self._x_local_seq_len: Optional[int] = None
        self._use_full_unified_sequence: Optional[bool] = None
        self._num_replicated_suffix: Optional[int] = None
        self._patch_size: Optional[int] = None
        self._f_patch_size: Optional[int] = None

        # Reference to the model (set at capture time)
        self._transformer = None

        # cache-dit state: cache_context from the CachedBlocks wrapper
        # (needed to call set_context() before can_cache())
        self._cache_context = None

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
        Phase 2: Capture each segment sequentially.
        """
        self._transformer = dit_model
        self._patch_size = static_kwargs.get("patch_size", 2)
        self._f_patch_size = static_kwargs.get("f_patch_size", 1)

        # Extract cache_context for calling set_context() before can_cache().
        # cache-dit's CachedContextManager stores CachedContext objects in its
        # _cached_context_manager dict. We need one to pass to set_context().
        # Note: cache-dit does NOT replace dit_model.layers — it creates a
        # separate CachedBlocks wrapper object. The context_manager and
        # cache_context are accessible via transformer._context_manager.
        ctx_mgr = getattr(dit_model, "_context_manager", None)
        if ctx_mgr is not None:
            registry = getattr(ctx_mgr, "_cached_context_manager", {})
            if registry:
                # Get the first (and typically only) CachedContext
                for ctx_name, ctx_obj in registry.items():
                    self._cache_context = ctx_obj
                    logger.info(
                        "  Found cache_context '%s' from context_manager registry",
                        ctx_name,
                    )
                    break
            else:
                logger.warning(
                    "  context_manager found but _cached_context_manager is empty"
                )
        else:
            logger.info("  No _context_manager on transformer (cache-dit not enabled)")

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
        # Buffer to save Fn block INPUT (before Fn execution) for residual calculation
        self._fn_input_snapshot = torch.empty_like(pre_out.unified)

        # --- Phase 2: Capture each segment ---
        logger.info("StepLevel CUDA Graph: Phase 2 — capturing segments")

        nrs = self._num_replicated_suffix
        ufs = self._use_full_unified_sequence

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
        """Copy new per-request data into persistent static buffers."""
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
            step_is_all_cached: True if SCM says this step is all-cache.
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
            # Replicate CachedBlocks_Pattern_3_4_5.forward() sequence:
            #   1. set_context()     ← before Fn blocks
            #   2. Fn blocks execute ← graph_fn.replay()
            #   3. mark_step_begin() ← after Fn, before can_cache
            #   4. can_cache()       ← decision
            self._set_cache_context()

            # Save Fn input BEFORE graph_fn executes (for residual calculation)
            self._fn_input_snapshot.copy_(self.inter_buffer)

            self.graph_fn.replay()

            # Eager: query DBCache decision
            step_can_cache = self._query_dbcache_decision()

            if step_can_cache:
                self.inter_buffer.copy_(self.cached_outputs)
            else:
                self.graph_middle_full.replay()
                self.cached_outputs.copy_(self.inter_buffer)

            # Bn blocks (only if Bn > 0)
            if self.graph_bn is not None:
                self.graph_bn.replay()

        # Post-block (every step)
        self.graph_post.replay()

        return self.output_buffer

    def _set_cache_context(self):
        """Set cache-dit context at the start of each step (before Fn blocks).

        Mirrors CachedBlocks.forward() step 1: set_context().
        Must be called before graph_fn.replay() and _query_dbcache_decision().
        """
        ctx_mgr = getattr(self._transformer, "_context_manager", None)
        if ctx_mgr is None:
            return

        # Lazy-initialize _cache_context if not set at capture time.
        if self._cache_context is None:
            registry = getattr(ctx_mgr, "_cached_context_manager", {})
            if registry:
                for ctx_name, ctx_obj in registry.items():
                    self._cache_context = ctx_obj
                    logger.info(
                        "  Lazy-initialized cache_context '%s' from registry",
                        ctx_name,
                    )
                    break

        if self._cache_context is not None:
            ctx_mgr.set_context(self._cache_context)

    def _query_dbcache_decision(self) -> bool:
        """Call cache-dit's can_cache() in eager mode after Fn blocks replay.

        Returns True if remaining blocks can use cache.

        Call sequence (mirroring CachedBlocks_Pattern_3_4_5.forward()):
          1. _set_cache_context()  ← called by replay_step before graph_fn
          2. save _fn_input_snapshot ← inter_buffer before Fn
          3. graph_fn.replay()     ← Fn blocks execute, inter_buffer updated
          4. compute residual = inter_buffer - _fn_input_snapshot
          5. mark_step_begin()     ← after Fn, before can_cache
          6. can_cache(residual)   ← decision based on Fn RESIDUAL, not output
          7. if cache miss: set_Fn_buffer(residual) ← save for next step
        """
        # Compute Fn residual: fn_output - fn_input
        # This is what CachedBlocks._get_Fn_residual() does
        fn_residual = self.inter_buffer - self._fn_input_snapshot
        # Save Fn output snapshot for cached_outputs update
        self._fn_output_snapshot.copy_(self.inter_buffer)

        ctx_mgr = getattr(self._transformer, "_context_manager", None)
        if ctx_mgr is None:
            return False

        # mark_step_begin() AFTER Fn blocks, BEFORE can_cache()
        ctx_mgr.mark_step_begin()

        # Pass RESIDUAL (not raw output) to can_cache — this is what
        # CachedBlocks.forward() does via _get_Fn_residual()
        result = ctx_mgr.can_cache(fn_residual)

        if not result:
            # Cache miss: save Fn residual as baseline for next step's
            # similarity comparison. This is what CachedBlocks.forward()
            # does: ctx_mgr.set_Fn_buffer(Fn_hidden_states_residual)
            ctx_mgr.set_Fn_buffer(fn_residual)
        else:
            # Cache hit: increment cached step counter.
            # Mirrors CachedBlocks.forward(): ctx_mgr.add_cached_step()
            ctx_mgr.add_cached_step()

        return result

    def _update_fn_wrapper_cached_output(self):
        """No longer needed — set_Fn_buffer() is called directly in
        _query_dbcache_decision(). Kept as placeholder for potential
        future Bn buffer updates."""
        pass

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
        self._fn_input_snapshot = None
        self._transformer = None
        self._cache_context = None
        self._captured = False
