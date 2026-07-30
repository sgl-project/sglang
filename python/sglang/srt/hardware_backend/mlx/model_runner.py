"""MLX model runner for Apple Silicon.

Slot allocation and radix-trie prefix matching are handled by the
scheduler (``TokenToKVPoolAllocator`` / ``RadixCache``).  This runner
reads cached attention KV from ``MlxAttentionKVPool``, restores any
native auxiliary layer state, runs the forward pass, and writes the new
cache state back.  Each request keeps model-shaped cache entries:
attention layers use ``ContiguousAttentionKVCache`` and auxiliary layers
use native ``mlx-lm`` cache objects.

The module also exposes a lazy-eval (`*_start` / `*_finalize`) surface
used by the MLX overlap scheduler to pipeline CPU bookkeeping with
GPU execution.  The lazy API is a thin split of the synchronous API:
``*_start`` builds the compute graph without materialising outputs,
``*_finalize`` blocks on the lazy token(s) and commits per-request
state.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import psutil
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_lm_load
from mlx_lm.utils import quantize_model as mlx_lm_quantize_model

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.aot import (
    MLX_AOT_KERNEL_REGISTRY,
    MlxAOTKernelSet,
)
from sglang.srt.hardware_backend.mlx.kv_cache import (
    AttentionOffsetCache,
    BatchedDecodeContext,
    ContiguousAttentionKVCache,
    MlxAttentionKVPool,
    MLXAttentionWrapper,
    MlxModelCacheLayout,
    PoolBackedAttentionKVCache,
    clear_context,
    find_attention_layers,
    get_head_dim,
    get_num_heads,
    get_num_kv_heads,
    patch_model_attention,
    set_context,
    uses_sliding_window_attention,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_server_args

logger = logging.getLogger(__name__)


@dataclass
class MlxPendingPrefill:
    """Lazy prefill state, finalised after ``mx.eval``/``async_eval``.

    ``cache`` is the per-layer cache list that will
    become ``_req_caches[req_id]`` once the request is committed.  It
    may have been converted from transient pool-backed attention caches
    already, so its ``state`` arrays are safe to hand to ``async_eval``.
    """

    lazy_token: mx.array
    cache: list[Any]
    req_id: str
    full_token_ids: list[int]
    req_pool_idx: int
    synced_offset: int


@dataclass
class MlxPendingExtend:
    """Lazy chunked-prefill-continuation state for an existing request.

    Mirrors :meth:`MlxModelRunner.extend` split into launch/finalize
    halves.  ``cache`` is the request's existing per-layer cache (not a
    fresh one) so the graph writes extend onto the already-materialised
    prefix.
    """

    lazy_token: mx.array
    req_id: str
    new_token_ids: list[int]
    new_synced_offset: int


@dataclass
class MlxPendingDecode:
    """Lazy decode state, finalised after ``mx.eval``/``async_eval``.

    ``caches`` is a per-request list of per-layer cache
    references (``caches[req_idx][layer_idx]``).  These are the same
    objects the attention wrapper writes into during the forward pass,
    so :meth:`decode_batch_start_chained` can launch the next step on
    top of the same caches without materialising this step first.
    """

    lazy_tokens: mx.array
    req_ids: list[str]
    caches: list[list[Any]]


_MLX_QUANTIZATION_PRESETS: dict[str, tuple[int, int]] = {
    # name -> (bits, group_size). group_size=64 matches the mlx-community convention.
    "mlx_q4": (4, 64),
    "mlx_q8": (8, 64),
}
_MLX_KV_FLOAT_DTYPES = {mx.float16, mx.bfloat16, mx.float32}

# Memory-safe chunked prefill on Apple Silicon. A long prefill whose activation peak
# exceeds the Metal working set aborts the scheduler with an uncatchable command-buffer
# OOM, so the chunk is sized to keep that peak below the limit.
#
# These are conservative startup estimates, not tuned constants: the in-server probe
# (_calibrate_prefill_chunk) measures the real per-token transient at runtime and revises
# the chunk down before serving, so they only have to over-estimate safely. Baselines come
# from the original repro -- M5 Pro 24GB, Qwen3-30B-A3B-4bit, radix off: 17.76 GiB working
# set, ~17.12 GiB steady-state baseline after the first prefill; a 256-token chunk peaked
# at 17.43 GiB and held while a 398-token chunk peaked at 17.58 GiB and crashed (the limit
# is soft, so a sub-limit peak can still abort). How the chunk-sizing values below feed
# _safe_prefill_chunk_tokens (chunk = headroom * SAFETY / (ACT_BYTES * n_q_heads * ctx)):
#   ACT_BYTES_PER_QHEAD -- conservative estimate of the (chunk x ctx) attention
#     scores/softmax bytes per q-head; the probe overwrites it with the measured cost.
#   SAFETY -- commit only 60% of headroom to the transient; in the repro ~0.72 of headroom
#     crashed while ~0.48 held, so 0.6 stays in the stable band with margin.
#   MIN_PREFILL_CHUNK -- chunk floor: below this prefill is too slow to serve, so fail
#     loudly. Also inverted in _memory_safe_max_running so the cap and chunk stay aligned.
_MLX_PREFILL_ACT_BYTES_PER_QHEAD = 32  # conservative per-(token x ctx) cost per q-head
_MLX_PREFILL_SAFETY = 0.6  # fraction of headroom the chunk's transient may use
_MLX_MIN_PREFILL_CHUNK = 256  # chunk floor (tokens); see block comment above
# Minimum KV-pool size in tokens, so the pool never collapses to something unusable (the
# radix-off no-context-length branch and the radix-on auto-sizer). A pool floor, distinct
# from the equal-valued _MLX_MIN_PREFILL_CHUNK (a chunk floor).
_MLX_MIN_POOL_SLOTS = 256
# Cap the KV pool at context x running x this (25% slack), not greedily.
_MLX_POOL_SLOTS_SLACK = 1.25
# Startup probe (_calibrate_prefill_chunk): replay a small prefill to measure the real
# resident baseline + transient, then revise the chunk down. PROBE_CHUNK / PROBE_MAX_CONTEXT
# keep the replay small so it never nears the edge yet still shows the linear-in-ctx trend.
# PROBE_FLOOR_CHUNK is the post-measurement floor: having measured the true cost the probe
# may trust a chunk below the conservative MIN_PREFILL_CHUNK, but below this prefill is
# impractical (fail loudly). PROBE_HEADROOM_BYTES is the live margin kept below the limit
# while probing, so the probe never triggers the OOM it measures for. PIPELINE_CACHE_SLACK
# is the extra private caches the overlap pipeline / pool keep resident beyond the running
# set (in-flight next step + pooled finished).
_MLX_PROBE_CHUNK = 128
_MLX_PROBE_MAX_CONTEXT = 2048
_MLX_PROBE_FLOOR_CHUNK = 32
_MLX_PROBE_HEADROOM_BYTES = 256 * 1024 * 1024
_MLX_PIPELINE_CACHE_SLACK = 2


class MlxModelRunner:
    """MLX model runner with radix-cache prefix sharing."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = False,
        pool_size: int | None = None,
        mem_fraction_static: float = 0.8,
        quantization: str | None = None,
        context_length: int | None = None,
        max_running_requests: int | None = None,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.disable_radix_cache = disable_radix_cache
        self._mem_fraction_static = mem_fraction_static
        # Used to size the KV pool and a safe prefill chunk; None skips chunk sizing.
        self._context_length = context_length
        self._max_running_requests = max_running_requests
        # Safe prefill chunk cap and its per-(token x ctx) cost; set during sizing/probe
        # and reused by the runtime admission gate.
        self._max_safe_prefill_chunk: int | None = None
        self._prefill_act_per_tok_ctx: int = 0
        # Memory-safe scheduler concurrency cap (read by the stub) and the per-request
        # resident cache cost (radix off). Set in _compute_pool_size; the cache cost is
        # refined by the probe. A None cap means the stub's pool-derived heuristic decides
        # (radix-on / no-context path, where private caches are not the resident cost).
        self._effective_max_running: int | None = None
        self._radix_off_cache_bytes: int = 0
        # Counter used to trigger periodic mx.clear_cache() calls.
        self._decode_step_ct: int = 0
        self._clear_steps = envs.SGLANG_MLX_CLEAR_CACHE_STEPS.get()
        # On-the-fly quantization preset (e.g. "mlx_q4"). None = no on-load quantization.
        # Pre-quantized HF repos load correctly regardless of this setting:
        # mlx_lm.load() detects the config and instantiates QuantizedLinear
        # modules directly.
        self._quantization: str | None = quantization

        self._load_model()

        # Pin MLX allocations to prevent OS paging
        device_info = mx.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            mx.set_wired_limit(max_wired)
            logger.info(f"Wired memory limit set to {max_wired / (1024**3):.1f} GB")

        patch_model_attention(self.model)

        layer_list, attn_attrs = find_attention_layers(self.model)
        self._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            layer_list,
            attn_attrs,
        )
        if self._cache_layout.num_attention_layers == 0:
            raise RuntimeError("MLX model has no supported attention layers")
        if self._cache_layout.has_auxiliary_state and not hasattr(
            self.model, "make_cache"
        ):
            raise RuntimeError(
                "MLX models with auxiliary cache state require model.make_cache()."
            )
        if self._cache_layout.has_auxiliary_state:
            self._model_embed, self._model_norm, self._model_lm_head = (
                self._extract_model_components()
            )
        self._max_seq_len = 4096  # doubles on overflow

        self._req_caches: dict[str, list[Any]] = {}
        self._req_token_ids: dict[str, list[int]] = {}
        self._cache_pool: list[list[Any]] = []  # reusable full-attention caches

        self._attention_kv_pool: MlxAttentionKVPool | None = None
        self._req_to_token_pool: ReqToTokenPool | None = None
        self._req_pool_idx: dict[str, int] = {}
        self._req_synced_offset: dict[str, int] = {}

        self._pool_size = self._compute_pool_size(pool_size)
        self._aot_kernels = self._build_aot_kernels()
        # Refine the chunk cap with a real prefill measurement, before the worker reads it.
        self._calibrate_prefill_chunk()

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _new_cache_skeleton(self) -> list[Any]:
        """Create a model-shaped cache list before attention cache wiring."""
        if self._cache_layout.has_auxiliary_state:
            cache = self.model.make_cache()
            if len(cache) != self._cache_layout.num_layers:
                raise RuntimeError(
                    "model.make_cache() returned "
                    f"{len(cache)} entries for {self._cache_layout.num_layers} layers"
                )
        else:
            cache = [None] * self._cache_layout.num_layers
        return cache

    def _new_native_cache(self) -> list[Any]:
        """Create a model-shaped cache list with attention KV adapters."""
        cache = self._new_cache_skeleton()
        for layer_idx in self._cache_layout.attention_layer_indices:
            cache[layer_idx] = ContiguousAttentionKVCache(max_seq_len=self._max_seq_len)
        return cache

    def _acquire_cache(self) -> list[Any]:
        """Get a reusable cache list from the pool, or create a new one."""
        if not self._cache_layout.has_auxiliary_state and self._cache_pool:
            cache = self._cache_pool.pop()
            for c in cache:
                c.offset = 0
            return cache
        return self._new_native_cache()

    def _release_cache(self, cache: list[Any]) -> None:
        """Return a cache list to the pool for reuse."""
        if not self._cache_layout.has_auxiliary_state:
            self._cache_pool.append(cache)

    def _first_attention_cache(self, cache: list[Any]) -> Any:
        return cache[self._cache_layout.first_attention_layer_index]

    def _get_auxiliary_state_pool_index(self, req_pool_idx: int) -> Any | None:
        if (
            not self._cache_layout.has_auxiliary_state
            or self._req_to_token_pool is None
            or not hasattr(self._req_to_token_pool, "get_auxiliary_state_indices")
        ):
            return None
        return self._req_to_token_pool.get_auxiliary_state_indices(req_pool_idx)

    def _get_auxiliary_state_pool(self) -> Any | None:
        return getattr(self._req_to_token_pool, "auxiliary_state_pool", None)

    def _restore_auxiliary_state(self, req_pool_idx: int, cache: list[Any]) -> bool:
        pool_index = self._get_auxiliary_state_pool_index(req_pool_idx)
        pool = self._get_auxiliary_state_pool()
        if pool_index is None or not hasattr(pool, "restore_cache"):
            return False
        return pool.restore_cache(
            pool_index,
            cache,
            self._cache_layout.auxiliary_layer_indices,
        )

    def _store_auxiliary_state(self, req_pool_idx: int, cache: list[Any]) -> None:
        pool_index = self._get_auxiliary_state_pool_index(req_pool_idx)
        pool = self._get_auxiliary_state_pool()
        if pool_index is None or not hasattr(pool, "store_cache"):
            return
        pool.store_cache(
            pool_index,
            cache,
            self._cache_layout.auxiliary_layer_indices,
        )

    def store_auxiliary_state_for_request(self, req_id: str) -> None:
        """Snapshot native auxiliary state before scheduler-owned radix insert."""
        req_pool_idx = self._req_pool_idx.get(req_id)
        cache = self._req_caches.get(req_id)
        if req_pool_idx is None or cache is None:
            return
        self._store_auxiliary_state(req_pool_idx, cache)

    def _select_auxiliary_state_track_len(
        self,
        *,
        prefix_len: int,
        new_token_count: int,
        full_len: int,
        req: Any | None,
    ) -> int | None:
        if (
            not self._cache_layout.has_auxiliary_state
            or req is None
            or new_token_count <= 0
        ):
            return None

        chunk_size = get_server_args().mamba_cache_chunk_size
        track_len = prefix_len + (new_token_count // chunk_size) * chunk_size
        branching_len = getattr(req, "mamba_branching_seqlen", None)
        if (
            branching_len is not None
            and prefix_len < branching_len <= prefix_len + new_token_count
            and (branching_len - prefix_len) % chunk_size == 0
        ):
            track_len = branching_len

        if track_len <= prefix_len or track_len > full_len:
            return None
        return track_len

    def _store_tracked_auxiliary_state(
        self,
        req: Any | None,
        cache: list[Any],
        track_len: int | None,
    ) -> None:
        if (
            req is None
            or track_len is None
            or not self._cache_layout.has_auxiliary_state
        ):
            return
        pool = self._get_auxiliary_state_pool()
        if pool is None or not hasattr(pool, "store_cache"):
            return

        track_buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
        if track_buffer is None:
            track_buffer = pool.alloc(1)
            if track_buffer is None:
                logger.warning(
                    "MLX auxiliary-state track slot allocation failed; "
                    "falling back to leaf-only auxiliary-state radix caching."
                )
                return
            req.mamba_ping_pong_track_buffer = track_buffer
            req.mamba_next_track_idx = 0

        pool.store_cache(
            track_buffer[0],
            cache,
            self._cache_layout.auxiliary_layer_indices,
        )
        req.mamba_last_track_seqlen = track_len

    def _cache_with_pool_backed_attention(
        self, prefix_slot_ids: list[int], prefix_len: int
    ) -> list[Any]:
        assert self._attention_kv_pool is not None
        slot_ids_mx = mx.array(prefix_slot_ids, dtype=mx.int32)
        cache = self._new_cache_skeleton()
        for layer_idx in self._cache_layout.attention_layer_indices:
            cache[layer_idx] = PoolBackedAttentionKVCache(
                self._attention_kv_pool,
                self._cache_layout.attention_pool_index(layer_idx),
                slot_ids_mx,
                prefix_len,
            )
        return cache

    def _materialize_pool_backed_attention(self, cache: list[Any]) -> list[Any]:
        contiguous_cache = self._acquire_cache()
        for layer_idx in self._cache_layout.attention_layer_indices:
            pbc = cache[layer_idx]
            contiguous_cache[layer_idx].update_and_fetch(
                pbc._full_keys, pbc._full_values
            )
        for layer_idx in self._cache_layout.auxiliary_layer_indices:
            contiguous_cache[layer_idx] = cache[layer_idx]
        return contiguous_cache

    @staticmethod
    def _cache_arrays(cache: Any) -> list[mx.array]:
        """Return every MLX array nested under ``cache.state``."""
        arrays: list[mx.array] = []

        def collect(value: Any) -> None:
            if isinstance(value, mx.array):
                arrays.append(value)
            elif value is None:
                return
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(getattr(cache, "state", ()))
        return arrays

    @staticmethod
    def _eval_with_cache(token_result: mx.array, cache: list[Any]) -> None:
        """Evaluate token result and all cache buffers in one mx.eval call."""
        mx.eval(
            token_result,
            *[s for c in cache for s in MlxModelRunner._cache_arrays(c)],
        )

    @staticmethod
    def _cache_state_arrays(pending_caches: list[list[Any]]) -> list[mx.array]:
        """Flatten pending decode cache state list into an array list.

        Safe to hand to ``mx.async_eval``.
        """
        return [
            s
            for cache_list in pending_caches
            for cache in cache_list
            for s in MlxModelRunner._cache_arrays(cache)
        ]

    def _load_model(self):
        """Load model using mlx_lm. If ``self._quantization`` requests a preset
        (e.g. ``mlx_q4``), quantize fp16 weights in-place via
        :func:`mlx_lm.utils.quantize_model` after load.
        """
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        # We need the config dict to pass into quantize_model so it knows tied/embedding
        # layout. return_config=True is cheap and ignored when no quantization is requested.
        loaded = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
            return_config=True,
        )
        self.model, _tokenizer, config = loaded

        if self._quantization in _MLX_QUANTIZATION_PRESETS:
            bits, group_size = _MLX_QUANTIZATION_PRESETS[self._quantization]
            # Skip if the model was already loaded quantized (pre-quantized HF repo);
            # mlx_lm.load detects the config and instantiates QuantizedLinear directly,
            # so applying the preset on top would be redundant.
            if "quantization" in (config or {}):
                logger.info(
                    "MLX model is already quantized by the HF repo; "
                    f"ignoring --quantization={self._quantization}"
                )
            else:
                # Read weight-tensor totals from MLX array metadata (shape + dtype).
                # This is zero-cost — neither materializes the lazy fp16 weights nor
                # forces them to be peak-resident in memory at once (which on a 64 GB
                # Mac running a 32 B model would put us within a few GB of OOM).
                bytes_before = sum(
                    p.size * p.itemsize
                    for _, p in tree_flatten(self.model.parameters())
                )
                q_start = time.time()
                logger.info(
                    f"Quantizing MLX model on-the-fly: bits={bits} "
                    f"group_size={group_size} (preset={self._quantization})"
                )
                self.model, _new_config = mlx_lm_quantize_model(
                    self.model,
                    config or {},
                    group_size=group_size,
                    bits=bits,
                )
                bytes_after = sum(
                    p.size * p.itemsize
                    for _, p in tree_flatten(self.model.parameters())
                )
                q_time = time.time() - q_start
                pct_reduction = (1 - bytes_after / max(bytes_before, 1)) * 100
                logger.info(
                    f"Quantization complete in {q_time:.2f}s — "
                    f"weight bytes: {bytes_before / 1024**3:.2f} GB -> "
                    f"{bytes_after / 1024**3:.2f} GB ({pct_reduction:.1f}% reduction)"
                )

        # Force-evaluate weights so mx.get_active_memory() reflects
        # actual usage before attention KV pool sizing.
        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

        # Optional: Path B fusion — keep up_proj/gate_proj weights separate
        # (no matmul-kernel tile regression) but fuse the swiglu activation
        # into the gate matmul via a custom Metal kernel. Activated by
        # SGLANG_MLX_FUSE_SWIGLU=1. Mutually exclusive with FUSE_SWITCHGLU.
        # See: python/sglang/srt/hardware_backend/mlx/moe/fused_swiglu.py
        if envs.SGLANG_MLX_FUSE_SWIGLU.get():
            from sglang.srt.hardware_backend.mlx.moe.fused_swiglu import (
                patch_switch_glu_with_fused_swiglu,
            )

            n_patched = patch_switch_glu_with_fused_swiglu(self.model)
            logger.info(
                f"MLX SwiGLU activation fusion enabled: patched {n_patched} blocks"
            )

    def _attention_module_for_layer(self, layer_idx: int) -> Any:
        attn = getattr(
            self._cache_layout.layers[layer_idx],
            self._cache_layout.attention_attr(layer_idx),
        )
        if isinstance(attn, MLXAttentionWrapper):
            return attn._inner
        return attn

    def _attention_kv_config_for_layer(
        self, layer_idx: int
    ) -> tuple[int, int, mx.Dtype]:
        layer = self._cache_layout.layers[layer_idx]
        sample_attn = self._attention_module_for_layer(layer_idx)
        if uses_sliding_window_attention(layer, sample_attn):
            raise NotImplementedError(
                "MLX radix attention KV pool does not support sliding-window "
                f"attention yet at layer {layer_idx}. Sliding-window KV needs "
                "per-layer/window-aware pools."
            )
        n_kv_heads = get_num_kv_heads(sample_attn)
        if n_kv_heads is None:
            raise RuntimeError(
                f"Cannot determine n_kv_heads from attention module at layer {layer_idx}"
            )
        head_dim = get_head_dim(sample_attn)
        if head_dim is None:
            raise RuntimeError(
                f"Cannot determine head_dim from attention module at layer {layer_idx}"
            )
        dtype = mx.float16
        if hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            dtype = sample_attn.k_proj.weight.dtype
        if dtype not in _MLX_KV_FLOAT_DTYPES:
            # QuantizedLinear packs weights as integers, but the KV cache
            # stores dequantized projection outputs, which are produced in
            # the compute dtype carried by the quantization scales.  Storing
            # at that dtype instead of float32 halves pool bytes per slot
            # and keeps prefix-hit forwards in the same dtype as the no-hit
            # path (a float32 pool promoted every post-hit concat).
            scales = getattr(sample_attn.k_proj, "scales", None)
            if scales is not None and scales.dtype in _MLX_KV_FLOAT_DTYPES:
                dtype = scales.dtype
            else:
                dtype = mx.float32
        return n_kv_heads, head_dim, dtype

    def _get_attn_config(self) -> tuple[int, int, mx.Dtype]:
        """Return the uniform attention KV config used by the shared MLX pool."""
        if self._cache_layout.num_attention_layers == 0:
            raise RuntimeError(
                "Cannot determine attention config: no attention module found"
            )
        first_layer_idx = self._cache_layout.first_attention_layer_index
        first_config = self._attention_kv_config_for_layer(first_layer_idx)
        for layer_idx in self._cache_layout.attention_layer_indices[1:]:
            config = self._attention_kv_config_for_layer(layer_idx)
            if config != first_config:
                raise NotImplementedError(
                    "MLX radix attention KV pool requires uniform softmax-attention "
                    "KV shape across layers. "
                    f"Layer {first_layer_idx} has {first_config}, "
                    f"but layer {layer_idx} has {config}. "
                    "Heterogeneous attention KV or sliding-window KV needs "
                    "per-layer pools."
                )
        return first_config

    def _compute_pool_size(self, explicit_size: int | None) -> int:
        """Size the KV pool and (radix off only) a memory-safe prefill chunk cap.

        Radix off (the default MLX config and the path this prefill-OOM fix targets):
        clamp the budget to real free memory, cap the pool at the running set's context
        need so the rest stays as prefill headroom, and derive a chunk cap that keeps the
        prefill below the Metal working-set edge; raise if even a minimum prefill cannot
        fit. The startup chunk is a conservative estimate; the probe refines it.

        Radix on: a single greedy, pre-allocated shared pool backs prefix reuse, so the
        radix-off private-cache resident model below does not apply -- modelling the whole
        greedy pool as coexisting with the prefill would structurally starve the headroom
        and wrongly refuse startup. Keep the prior simple auto-sizing and leave the chunk
        uncapped (admission gate inactive), so radix-on behaviour is unchanged by this fix.
        """
        if not self.disable_radix_cache:
            # See docstring: the chunk-sizing model is radix-off-specific. Leave the cap
            # unset (no worker cap, no admission gate) and keep prior pool auto-sizing.
            self._max_safe_prefill_chunk = None
            self._prefill_act_per_tok_ctx = 0
            # None concurrency cap: defer to the stub's pool-derived heuristic (the radix-on
            # shared pool, not per-request private caches, governs its memory profile).
            self._effective_max_running = None
            return self._auto_pool_size(explicit_size)

        n_kv_heads, head_dim, dtype = self._get_attn_config()
        first_attn = self._attention_module_for_layer(
            self._cache_layout.first_attention_layer_index
        )
        n_q_heads = get_num_heads(first_attn) or n_kv_heads
        num_layers = self._cache_layout.num_attention_layers
        sys_available = psutil.virtual_memory().available
        mlx_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        mlx_used = mx.get_active_memory()
        gib = 1024**3
        context_length = self._context_length
        bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * dtype.size

        # Headroom above weights, clamped to real free memory (working-set limit alone is
        # only a no-pressure recommendation). Shared with the admission gate.
        dynamic_budget = self._gpu_headroom_bytes(mlx_limit, mlx_used, sys_available)
        mem_fraction_budget = max(
            int(mlx_limit * self._mem_fraction_static) - mlx_used, 0
        )
        kv_budget = min(dynamic_budget, mem_fraction_budget)

        # Conservative startup estimate of the per-(token x ctx) activation cost; the probe
        # overwrites it with the measured value for the admission gate.
        per_tok_ctx = _MLX_PREFILL_ACT_BYTES_PER_QHEAD * n_q_heads
        self._prefill_act_per_tok_ctx = per_tok_ctx

        # Radix off: each request owns a private ContiguousAttentionKVCache pre-allocated to
        # its full power-of-two span, so its resident cost is span x bytes_per_slot,
        # independent of the prompt length. Record it for the resident model, the concurrency
        # cap, and the admission gate; the probe refines it with a real measurement.
        if context_length is not None:
            cache_tokens = self._max_seq_len
            while cache_tokens < context_length:
                cache_tokens *= 2
        else:
            cache_tokens = self._max_seq_len
        per_cache_bytes = cache_tokens * bytes_per_slot
        self._radix_off_cache_bytes = per_cache_bytes

        # Scheduler concurrency cap: honor an explicit --max-running-requests, else default to
        # the most requests whose private caches still leave a usable prefill. This becomes the
        # scheduler's real running cap (the stub reads it) and the chunk sizing below assumes
        # it, so capping concurrency here is what keeps coexisting caches from overflowing the
        # working set; the admission gate is the live-memory backstop on top.
        running = self._memory_safe_max_running(
            dynamic_budget, per_cache_bytes, per_tok_ctx, context_length
        )
        self._effective_max_running = running

        # Radix off: the pool can never exceed the running set's contexts, so cap it there
        # and leave the rest as prefill headroom.
        if context_length is not None:
            need_slots = int(context_length * running * _MLX_POOL_SLOTS_SLACK)
        else:
            need_slots = None

        if explicit_size is not None:
            pool_size = explicit_size
        else:
            pool_size = kv_budget // bytes_per_slot if bytes_per_slot else 0
            # need_slots is set iff context_length is, so these are mutually exclusive:
            # a known context caps the pool at its slot need, an unknown one only floors it.
            if need_slots is not None:
                pool_size = min(pool_size, need_slots)
            else:
                pool_size = max(pool_size, _MLX_MIN_POOL_SLOTS)

        # Resident KV the prefill coexists with: span x running private caches (see
        # per_cache_bytes above). Using pool_size here would undercount the baseline and
        # oversize the chunk.
        if context_length is not None:
            resident_kv_bytes = running * per_cache_bytes
        else:
            resident_kv_bytes = (pool_size + 1) * bytes_per_slot  # +1 padding slot 0
        activation_headroom = max(dynamic_budget - resident_kv_bytes, 0)
        safe_chunk = self._safe_prefill_chunk_tokens(
            activation_headroom, per_tok_ctx, context_length
        )

        if context_length is not None:
            if explicit_size is None and (
                pool_size < context_length or safe_chunk < _MLX_MIN_PREFILL_CHUNK
            ):
                remedies = []
                # Raising mem-fraction only helps when the pool itself is too small;
                # a starved prefill (pool ok but chunk too small) needs real free
                # memory, since activation headroom is drawn from the whole budget.
                needed_mf = (
                    (mlx_used + context_length * bytes_per_slot) / mlx_limit
                    if mlx_limit
                    else 1.0
                )
                if pool_size < context_length and needed_mf <= 0.99:
                    hint = (int(needed_mf * 100) + 1) / 100  # round up to 0.01
                    remedies.append(
                        f"raise --mem-fraction-static to ~{hint:.2f} (now "
                        f"{self._mem_fraction_static:.2f})"
                    )
                remedies.append(
                    "use a smaller / more-quantized model, a shorter --context-length, "
                    "or a machine with more unified memory"
                )
                remedy_text = "".join(f"    * {r}\n" for r in remedies)
                raise RuntimeError(
                    "MLX cannot fit both the KV cache and a safe prefill in the GPU "
                    "working set.\n"
                    f"  KV pool = {pool_size} tokens (need >= {context_length}); "
                    f"safe prefill chunk = {safe_chunk} tokens "
                    f"(need >= {_MLX_MIN_PREFILL_CHUNK}).\n"
                    f"  Budget (mem-fraction-static={self._mem_fraction_static:.2f}):\n"
                    f"    Metal working-set limit : {mlx_limit / gib:.2f} GiB\n"
                    f"    model weights resident  : {mlx_used / gib:.2f} GiB\n"
                    f"    free above weights      : {dynamic_budget / gib:.2f} GiB "
                    f"(system free {sys_available / gib:.2f} GiB)\n"
                    f"    KV cache resident       : {resident_kv_bytes / gib:.2f} GiB "
                    f"(leaves {activation_headroom / gib:.2f} GiB for prefill)\n"
                    f"    KV per token slot       : {bytes_per_slot} B "
                    f"({num_layers}L x {n_kv_heads}kv x {head_dim}d x 2 x {dtype.size}B)\n"
                    "  Options:\n"
                    f"{remedy_text}"
                    "  Or set --max-total-tokens explicitly (advanced)."
                )
            # Clamp to the context: a chunk larger than the context never chunks.
            self._max_safe_prefill_chunk = min(
                max(safe_chunk, _MLX_MIN_PREFILL_CHUNK), context_length
            )
        else:
            self._max_safe_prefill_chunk = None

        logger.info(
            f"Auto-sized attention KV pool: "
            f"sys_available={sys_available / gib:.2f} GB, "
            f"mlx_limit={mlx_limit / gib:.1f} GB, mlx_used={mlx_used / gib:.2f} GB, "
            f"dynamic_budget={dynamic_budget / gib:.2f} GB, "
            f"bytes_per_slot={bytes_per_slot}, pool_size={pool_size}, "
            f"resident_kv={resident_kv_bytes / gib:.2f} GB, "
            f"activation_headroom={activation_headroom / gib:.2f} GB, "
            f"act_cost={per_tok_ctx}B/tok-ctx, need_slots={need_slots}, "
            f"startup_safe_prefill_chunk={self._max_safe_prefill_chunk} "
            f"(refined by in-server probe)"
        )
        return pool_size

    def _auto_pool_size(self, explicit_size: int | None) -> int:
        """Prior radix-on pool sizing: auto-size from available memory, no chunk cap.

        Retained unchanged for the radix-on path (see ``_compute_pool_size``): the greedy
        shared pool's memory profile differs from the radix-off private caches the
        chunk-sizing model assumes, so radix-on keeps its original behaviour and is not
        gated by the prefill-chunk cap.
        """
        if explicit_size is not None:
            return explicit_size
        n_kv_heads, head_dim, dtype = self._get_attn_config()
        num_layers = self._cache_layout.num_attention_layers
        sys_available = psutil.virtual_memory().available
        mlx_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        mlx_used = mx.get_active_memory()
        mlx_usable = int(mlx_limit * self._mem_fraction_static)
        kv_budget = min(
            max(mlx_usable - mlx_used, 0),
            int(sys_available * self._mem_fraction_static),
        )
        bytes_per_slot = 2 * num_layers * n_kv_heads * head_dim * dtype.size
        pool_size = max(kv_budget // bytes_per_slot, _MLX_MIN_POOL_SLOTS)
        logger.info(
            f"Auto-sized attention KV pool (radix on): "
            f"sys_available={sys_available / (1024**3):.2f} GB, "
            f"mlx_limit={mlx_limit / (1024**3):.1f} GB, "
            f"mlx_used={mlx_used / (1024**3):.2f} GB, "
            f"kv_budget={kv_budget / (1024**3):.2f} GB, "
            f"bytes_per_slot={bytes_per_slot}, pool_size={pool_size}"
        )
        return pool_size

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def max_running_requests(self) -> int | None:
        """Memory-safe scheduler concurrency cap, read by the stub so the scheduler never
        admits more concurrent requests than the working set can hold. None means defer to
        the stub's pool-derived heuristic (radix-on / no auto-sizing)."""
        return self._effective_max_running

    @property
    def max_safe_prefill_chunk(self) -> int | None:
        """Largest prefill chunk (tokens) that keeps activations within the Metal
        working set, or None when not auto-derived (radix on, or no context length)."""
        return self._max_safe_prefill_chunk

    @staticmethod
    def _safe_prefill_chunk_tokens(
        activation_headroom_bytes: int,
        per_tok_ctx: int,
        context_length: int | None,
        safety: float = _MLX_PREFILL_SAFETY,
    ) -> int:
        """Largest chunk whose worst-case transient fits the activation headroom.

        The worst forward is the last chunk of a full-context prompt (its tokens attend to
        ~context_length resident), so transient ~= per_tok_ctx * chunk * context_length.
        Shared by the startup estimate and the probe.
        """
        per_token = per_tok_ctx * max(context_length or 0, 0)
        if per_token <= 0:
            return 0
        return int(max(activation_headroom_bytes, 0) * safety / per_token)

    def _memory_safe_max_running(
        self,
        dynamic_budget: int,
        per_cache_bytes: int,
        per_tok_ctx: int,
        context_length: int | None,
    ) -> int:
        """Resolve the radix-off scheduler concurrency cap.

        Honors an explicit ``--max-running-requests``. Otherwise defaults to the most
        requests whose private caches (plus the overlap pipeline's ``_MLX_PIPELINE_CACHE_SLACK``
        spare caches) still leave a ``_MLX_MIN_PREFILL_CHUNK`` prefill's activation headroom --
        the inverse of the chunk-sizing math, so the cap and the chunk stay consistent. Each
        radix-off cache is a fixed ``per_cache_bytes`` regardless of prompt length, so this is
        what stops many short concurrent requests from overflowing the working set. Floors at 1.
        """
        if self._max_running_requests is not None:
            return max(self._max_running_requests, 1)
        if context_length is None or per_cache_bytes <= 0 or per_tok_ctx <= 0:
            return 1
        # Activation a minimum useful chunk needs (inverse of _safe_prefill_chunk_tokens).
        act_min = (
            _MLX_MIN_PREFILL_CHUNK * per_tok_ctx * context_length / _MLX_PREFILL_SAFETY
        )
        fit = (
            int((dynamic_budget - act_min) // per_cache_bytes)
            - _MLX_PIPELINE_CACHE_SLACK
        )
        return max(fit, 1)

    @staticmethod
    def _gpu_headroom_bytes(mlx_limit: int, mlx_used: int, sys_available: int) -> int:
        """GPU bytes allocatable above current usage: the working-set limit, but never
        more than real free system memory (the limit is only a no-pressure recommendation,
        so budgeting against it alone lets a prefill overflow under external pressure).
        Shared by startup sizing and the runtime admission gate.
        """
        return max(min(mlx_limit, mlx_used + sys_available) - mlx_used, 0)

    def _live_prefill_headroom_bytes(self) -> int:
        """GPU headroom for prefill activations right now, from live device + system
        memory (reflects pressure that appeared after startup; live active already counts
        the resident KV)."""
        mlx_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        return self._gpu_headroom_bytes(
            mlx_limit, mx.get_active_memory(), psutil.virtual_memory().available
        )

    def _prefill_new_cache_bytes(self) -> int:
        """Bytes a newly admitted radix-off prefill will allocate for its private full-span
        cache. Zero when radix is on (shared pool) or when a finished cache can be reused (it
        is already resident, hence already counted in live active memory). For short prompts
        this fixed cost dominates the activation transient, so the admission gate must include
        it or it will admit a request whose cache allocation then overflows the working set.
        """
        if not self.disable_radix_cache or self._cache_pool:
            return 0
        return self._radix_off_cache_bytes

    def prefill_fits_live_memory(
        self, new_tokens: int, resident_ctx: int
    ) -> tuple[bool, int, int]:
        """Whether a prefill chunk fits the live GPU headroom, re-checked at the last
        instant before a forward so external pressure that eroded the startup margin turns
        into an up-front rejection instead of an uncatchable OOM. Uses the probe-measured
        per-(token x ctx) cost plus the fixed per-request cache the prefill will allocate.

        Returns ``(fits, estimate_bytes, headroom_bytes)``; admits when the cost is unknown
        (no calibration to act on).
        """
        per_tok_ctx = self._prefill_act_per_tok_ctx
        if per_tok_ctx <= 0:
            return True, 0, 0
        estimate = (
            per_tok_ctx * max(new_tokens, 0) * max(resident_ctx, 0)
            + self._prefill_new_cache_bytes()
        )
        headroom = self._live_prefill_headroom_bytes()
        return estimate <= headroom, estimate, headroom

    def live_safe_prefill_chunk(self) -> int | None:
        """Largest prefill chunk that fits live GPU headroom now (None if no auto-derived
        cap: radix on / uncalibrated). Same formula, measured cost, and full-context span as
        the startup cap and the gate, so it stays safe; exceeds the static cap only under
        light load. Floors at ``_MLX_MIN_PREFILL_CHUNK``, caps at ``context_length``.
        """
        context_length = self._context_length
        if (
            self._max_safe_prefill_chunk is None
            or self._prefill_act_per_tok_ctx <= 0
            or not context_length
        ):
            return None
        # TODO: a per-request prompt-length span would be tighter (needs per-request sizing).
        # Subtract the cache a fresh prefill allocates, like the admission gate does.
        headroom = self._live_prefill_headroom_bytes() - self._prefill_new_cache_bytes()
        dyn = self._safe_prefill_chunk_tokens(
            headroom, self._prefill_act_per_tok_ctx, context_length
        )
        return min(max(dyn, _MLX_MIN_PREFILL_CHUNK), context_length)

    def _calibrate_prefill_chunk(self) -> None:
        """Measure the real prefill footprint in-server and refine the chunk cap.

        Startup sizing snapshots memory before any forward runs, so it undercounts the
        steady-state resident KV: with radix off each request holds a private cache and
        finished caches stay pooled, so the resident set is the running set plus the
        overlap pipeline's caches (several hundred MiB the snapshot cannot see). That
        undercount is what lets the chunk grow large enough to overflow the working set.

        After load but before serving, allocate that steady-state cache set and read the
        resident baseline, then replay a chunked prefill on the last cache to read the
        worst single-forward peak. Size the chunk against the measured baseline (the fix)
        using the conservative cost (max of startup model and measured transient) so a
        baseline error cannot push the peak over the edge; store the measured transient
        for the admission gate so it does not over-reject.

        Safety: caches are materialised incrementally and stop if active nears the limit
        (the box cannot hold the resident set -> fail loudly); the replay uses a small
        chunk far from the edge; the wired limit is untouched; any measurement failure
        keeps the startup estimate (the probe degrades, never wedges startup).
        """
        context_length = self._context_length
        if (
            context_length is None
            or self.model is None
            or self._max_safe_prefill_chunk is None
        ):
            return  # no auto-derived chunk cap to refine (e.g. radix on / explicit pool)

        mlx_limit = mx.device_info().get(
            "max_recommended_working_set_size",
            mx.device_info().get("memory_size", 0),
        )
        if mlx_limit <= 0:
            logger.warning(
                "MLX prefill probe skipped: no device working-set limit reported; "
                "keeping startup chunk estimate %s.",
                self._max_safe_prefill_chunk,
            )
            return

        running = max(self._effective_max_running or 1, 1)
        target_caches = running + _MLX_PIPELINE_CACHE_SLACK
        chunk_probe = max(min(_MLX_PROBE_CHUNK, context_length), 1)
        ctx_target = min(context_length, _MLX_PROBE_MAX_CONTEXT)
        try:
            baseline, peak, ctx_reached, held = self._run_prefill_probe(
                chunk_probe, ctx_target, target_caches, mlx_limit
            )
        except Exception as exc:  # GPU / model error -> keep the startup estimate
            logger.warning(
                "MLX prefill probe failed (%s: %s); keeping startup chunk estimate %s.",
                type(exc).__name__,
                exc,
                self._max_safe_prefill_chunk,
            )
            return

        gib = 1024**3
        transient = max(peak - baseline, 0)
        score_cells = chunk_probe * max(ctx_reached, 1)
        # A real prefill must show a positive, non-trivial transient; if it does not, the
        # lazy graph likely was not forced -> distrust the measurement, keep the estimate.
        if transient < (1 << 20) or score_cells <= 0:
            logger.warning(
                "MLX prefill probe inconclusive (transient=%.3f GB over %d cells); "
                "keeping startup chunk estimate %s.",
                transient / gib,
                score_cells,
                self._max_safe_prefill_chunk,
            )
            return

        # Ceil; measuring at ctx_reached (<= context_length) over-estimates the full-context
        # cost, so extrapolation below stays conservative.
        measured_per_tok_ctx = -(-transient // score_cells)
        # Sizing uses the conservative cost; admission uses the measured one.
        startup_per_tok_ctx = self._prefill_act_per_tok_ctx
        sizing_per_tok_ctx = max(startup_per_tok_ctx, measured_per_tok_ctx)
        sys_available = psutil.virtual_memory().available
        headroom = self._gpu_headroom_bytes(mlx_limit, baseline, sys_available)
        safe = self._safe_prefill_chunk_tokens(
            headroom, sizing_per_tok_ctx, context_length
        )

        if held < target_caches or safe < _MLX_PROBE_FLOOR_CHUNK:
            raise RuntimeError(
                "MLX cannot serve this context: the measured resident footprint leaves "
                "too little GPU working set for even a minimum prefill.\n"
                f"  measured baseline ({held} resident KV cache(s) + runtime): "
                f"{baseline / gib:.2f} GiB\n"
                f"  Metal working-set limit                  : {mlx_limit / gib:.2f} GiB\n"
                f"  live headroom for prefill                : {headroom / gib:.2f} GiB\n"
                f"  largest safe prefill chunk               : {safe} tokens "
                f"(need >= {_MLX_PROBE_FLOOR_CHUNK}; held {held}/{target_caches} caches)\n"
                "  Use a smaller / more-quantized model, a shorter --context-length, or "
                "a machine with more unified memory."
            )

        startup_chunk = self._max_safe_prefill_chunk
        refined = min(max(safe, _MLX_PROBE_FLOOR_CHUNK), context_length)
        # Never relax above the conservative startup estimate (the probe grows only its
        # last cache, so it can under-count for context_length beyond the cache span).
        if startup_chunk is not None:
            refined = min(refined, startup_chunk)
        self._max_safe_prefill_chunk = refined
        self._prefill_act_per_tok_ctx = int(measured_per_tok_ctx)
        logger.info(
            "MLX prefill probe: baseline=%.3f GB (%d caches) peak=%.3f GB "
            "(chunk_probe=%d x ctx=%d) measured_act=%dB/tok-ctx "
            "(startup model %dB/tok-ctx; sizing used %dB) limit=%.2f GB headroom=%.2f GB; "
            "max_safe_prefill_chunk %s -> %d",
            baseline / gib,
            held,
            peak / gib,
            chunk_probe,
            ctx_reached,
            measured_per_tok_ctx,
            startup_per_tok_ctx,
            sizing_per_tok_ctx,
            mlx_limit / gib,
            headroom / gib,
            startup_chunk,
            refined,
        )

    def _run_prefill_probe(
        self, chunk_probe: int, ctx_target: int, target_caches: int, mlx_limit: int
    ) -> tuple[int, int, int, int]:
        """Measure the steady-state resident baseline and worst prefill transient.

        Allocates ``target_caches`` private caches (the resident KV the overlap scheduler /
        pool hold at steady state) via one-token forwards, then replays a chunked prefill
        on the last cache up to ``ctx_target`` for the worst single-forward peak. Returns
        ``(baseline, peak, ctx_reached, caches_held)``.

        Allocation stops if active reaches within ``_MLX_PROBE_HEADROOM_BYTES`` of the
        limit, so the probe never risks the OOM; ``caches_held < target_caches`` then means
        the box cannot hold the resident set (the caller fails loudly). Probe caches are
        discarded afterwards.
        """
        safe_ceiling = max(mlx_limit - _MLX_PROBE_HEADROOM_BYTES, 0)
        caches: list[list[Any]] = []

        def forward_chunk(cache: list[Any], start: int, n: int) -> None:
            # Varied (valid) ids so MoE routing exercises many experts, not just one.
            input_ids = mx.array(
                [[(start + i) % 1024 for i in range(n)]], dtype=mx.int32
            )
            logits = self._extract_logits(self.model(input_ids, cache=cache))
            self._eval_with_cache(mx.argmax(logits[:, -1, :], axis=-1), cache)

        try:
            mx.reset_peak_memory()
            weights_before = mx.get_active_memory()
            # 1. Materialise the steady-state resident cache set.
            for _ in range(target_caches):
                cache = self._new_native_cache()
                forward_chunk(cache, 0, 1)  # 1 token -> allocates the full-span cache
                caches.append(cache)
                if mx.get_active_memory() >= safe_ceiling:
                    break  # cannot hold the full resident set on this box
            baseline = mx.get_active_memory()
            # Refine the per-request cache cost from the real allocation (the analytic
            # bytes_per_slot is conservative); the admission gate uses the measured value.
            if self.disable_radix_cache and caches:
                measured_cache = int((baseline - weights_before) / len(caches))
                if measured_cache > 0:
                    self._radix_off_cache_bytes = measured_cache
            # 2. Worst single-forward transient: grow the last cache to ctx_target.
            last = caches[-1]
            mx.reset_peak_memory()
            offset = self._first_attention_cache(last).offset
            while offset < ctx_target and mx.get_peak_memory() < safe_ceiling:
                n = min(chunk_probe, ctx_target - offset)
                forward_chunk(last, offset, n)
                offset += n
            peak = max(mx.get_peak_memory(), baseline)
            return baseline, peak, offset, len(caches)
        finally:
            del caches
            mx.clear_cache()

    def _build_aot_kernels(self) -> MlxAOTKernelSet:
        """Build model-level set of optional registered AOT kernels."""
        if self._cache_layout.num_attention_layers == 0:
            return MlxAOTKernelSet()
        layer_idx = self._cache_layout.first_attention_layer_index
        sample_attn = getattr(
            self._cache_layout.layers[layer_idx],
            self._cache_layout.attention_attr(layer_idx),
        )
        n_kv_heads, head_dim, _ = self._get_attn_config()
        return MLX_AOT_KERNEL_REGISTRY.build_kernel_set(
            sample_attn=sample_attn,
            n_kv_heads=int(n_kv_heads),
            head_dim=int(head_dim),
        )

    def init_cache_pools(self, req_to_token_pool: ReqToTokenPool | None) -> None:
        """Create attention KV pool (+1 for padding slot 0)."""
        self._req_to_token_pool = req_to_token_pool
        if self.disable_radix_cache:
            return
        n_kv_heads, head_dim, dtype = self._get_attn_config()
        # +1 for padding slot 0
        self._attention_kv_pool = MlxAttentionKVPool(
            pool_size=self._pool_size + 1,
            num_layers=self._cache_layout.num_attention_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        logger.info(
            f"Attention KV pool initialized: pool_size={self._pool_size} "
            f"(buffer size {self._pool_size + 1} incl. padding slot 0), "
            f"{self._cache_layout.num_attention_layers} attention layers, "
            f"{n_kv_heads} kv_heads, {head_dim} head_dim"
        )

    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
    ) -> int:
        """Prefill a request.  Returns next_token_id."""
        pending = self.prefill_start(
            req_id=req_id,
            new_token_ids=new_token_ids,
            full_token_ids=full_token_ids,
            prefix_slot_ids=prefix_slot_ids,
            new_slot_ids=new_slot_ids,
            req_pool_idx=req_pool_idx,
            req=req,
        )
        self._eval_with_cache(pending.lazy_token, pending.cache)
        return self.prefill_finalize(pending)

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> int:
        """Continue prefill for a chunked request.  Returns next_token_id."""
        pending = self.extend_start(req_id, new_token_ids, new_slot_ids)
        self._eval_with_cache(pending.lazy_token, self._req_caches[req_id])
        return self.extend_finalize(pending)

    def _sync_new_kv_to_pool(
        self,
        cache: list[Any],
        cache_start: int,
        slot_ids: list[int],
    ) -> None:
        """Sync attention KV from contiguous cache to pool at the given slots."""
        if not slot_ids or self._attention_kv_pool is None:
            return
        end = cache_start + len(slot_ids)
        slot_ids_mx = mx.array(slot_ids, dtype=mx.int32)
        # TODO: Standardize ContiguousAttentionKVCache size to avoid transpose
        # Transpose cache (1, n_kv_heads, S, head_dim) to pool (S, n_kv_heads, head_dim)
        k_all = mx.stack(
            [
                cache[layer_idx].keys[0, :, cache_start:end, :].transpose(1, 0, 2)
                for layer_idx in self._cache_layout.attention_layer_indices
            ]
        )
        v_all = mx.stack(
            [
                cache[layer_idx].values[0, :, cache_start:end, :].transpose(1, 0, 2)
                for layer_idx in self._cache_layout.attention_layer_indices
            ]
        )
        self._attention_kv_pool.set_kv_all_layers(slot_ids_mx, k_all, v_all)

    def _sync_decode_kv_to_pool(self, req_id: str) -> None:
        """Sync un-flushed decode KV for *req_id* to the shared pool."""
        if self._attention_kv_pool is None or self._req_to_token_pool is None:
            return
        cache = self._req_caches.get(req_id)
        if cache is None:
            return
        current_offset = self._first_attention_cache(cache).offset
        synced_offset = self._req_synced_offset.get(req_id, 0)
        if current_offset <= synced_offset:
            return
        req_pool_idx = self._req_pool_idx.get(req_id)
        if req_pool_idx is None:
            return
        # Read slot IDs from scheduler's req_to_token_pool
        slot_ids = (
            self._req_to_token_pool.req_to_token[
                req_pool_idx, synced_offset:current_offset
            ]
            .to(dtype=int)
            .tolist()
        )
        self._sync_new_kv_to_pool(cache, synced_offset, slot_ids)
        self._req_synced_offset[req_id] = current_offset

    def flush_all_decode_kv(self) -> None:
        """Sync all active requests' un-flushed decode KV to the pool."""
        if self.disable_radix_cache or self._attention_kv_pool is None:
            return
        for req_id in list(self._req_caches.keys()):
            self._sync_decode_kv_to_pool(req_id)

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Decode one token per request."""
        pending = self.decode_batch_start(req_ids)
        # Evaluate lazy_tokens together with every affected cache buffer so
        # the attention write-then-read ordering is materialised in one
        # kernel submission.
        cache_arrays = self._cache_state_arrays(pending.caches)
        mx.eval(pending.lazy_tokens, *cache_arrays)
        return self.decode_batch_finalize(pending)

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
    ) -> MlxPendingPrefill:
        """Queue a prefill forward pass without evaluating.

        Returns an :class:`MlxPendingPrefill` containing the lazy
        next-token ``mx.array`` plus everything needed to commit the
        request in :meth:`prefill_finalize`.  The caller drives the GPU
        by handing ``lazy_token`` (and cache state) to ``mx.async_eval``.
        """
        prefix_len = len(prefix_slot_ids)
        if req is not None:
            req.mamba_last_track_seqlen = None

        if self.disable_radix_cache:
            cache = self._acquire_cache()
            input_ids = mx.array([new_token_ids], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            lazy_token = mx.argmax(logits[:, -1, :], axis=-1)
            return MlxPendingPrefill(
                lazy_token=lazy_token,
                cache=cache,
                req_id=req_id,
                full_token_ids=list(full_token_ids),
                req_pool_idx=req_pool_idx,
                synced_offset=0,
            )

        assert self._attention_kv_pool is not None

        new_token_count = len(new_token_ids)
        track_len = self._select_auxiliary_state_track_len(
            prefix_len=prefix_len,
            new_token_count=new_token_count,
            full_len=len(full_token_ids),
            req=req,
        )

        if prefix_len > 0:
            cache = self._cache_with_pool_backed_attention(prefix_slot_ids, prefix_len)
            pool_backed_attention = True
            restored_auxiliary_state = (
                not self._cache_layout.has_auxiliary_state
                or self._restore_auxiliary_state(req_pool_idx, cache)
            )
            if self._cache_layout.has_auxiliary_state and (
                not restored_auxiliary_state or new_token_count == 0
            ):
                # TODO(MLX): exact full-prefix hits need auxiliary state at
                # prefix_len - 1 to recompute last-token logits. The unified
                # tree stores state at the match boundary today, so use a
                # full-prompt fallback for that edge while still syncing newly
                # allocated attention KV below.
                cache = self._acquire_cache()
                input_ids = mx.array([full_token_ids or new_token_ids], dtype=mx.int32)
                model_output = self.model(input_ids, cache=cache)
                logits = self._extract_logits(model_output)
                lazy_token = mx.argmax(logits[:, -1, :], axis=-1)
                if new_slot_ids:
                    self._sync_new_kv_to_pool(cache, prefix_len, new_slot_ids)
                return MlxPendingPrefill(
                    lazy_token=lazy_token,
                    cache=cache,
                    req_id=req_id,
                    full_token_ids=list(full_token_ids),
                    req_pool_idx=req_pool_idx,
                    synced_offset=prefix_len + len(new_slot_ids),
                )
        else:
            cache = self._acquire_cache()
            pool_backed_attention = False

        if new_token_count > 0:
            track_new_count = track_len - prefix_len if track_len is not None else None
            if track_new_count is not None and 0 < track_new_count < new_token_count:
                input_ids = mx.array([new_token_ids[:track_new_count]], dtype=mx.int32)
                self.model(input_ids, cache=cache)
                self._store_tracked_auxiliary_state(req, cache, track_len)
                if pool_backed_attention:
                    cache = self._materialize_pool_backed_attention(cache)
                    pool_backed_attention = False
                extend_tokens = new_token_ids[track_new_count:]
            else:
                extend_tokens = new_token_ids
        else:
            # Full cache hit - rerun last token to get next-token logits
            extend_tokens = full_token_ids[-1:]
            for c in cache:
                c.offset = max(c.offset - 1, 0)

        input_ids = mx.array([extend_tokens], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        if track_len is not None and track_len == prefix_len + new_token_count:
            self._store_tracked_auxiliary_state(req, cache, track_len)

        last_logits = logits[:, -1, :]
        lazy_token = mx.argmax(last_logits, axis=-1)

        # Convert pool-backed attention KV to contiguous attention KV for decode.
        # This appends a lazy slice-assign onto the forward graph; the
        # arrays get materialised when the caller evaluates lazy_token.
        if pool_backed_attention:
            cache = self._materialize_pool_backed_attention(cache)

        if new_slot_ids:
            self._sync_new_kv_to_pool(cache, prefix_len, new_slot_ids)

        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=prefix_len + len(new_slot_ids),
        )

    def prefill_finalize(self, pending: MlxPendingPrefill) -> int:
        """Materialise a pending prefill and commit per-request state.

        Must be called *after* ``pending.lazy_token`` has been handed to
        ``mx.async_eval`` / ``mx.eval``.  ``.item()`` here is blocking on
        that specific lazy scalar.
        """
        next_token = int(pending.lazy_token.item())
        self._req_token_ids[pending.req_id] = list(pending.full_token_ids) + [
            next_token
        ]
        self._req_caches[pending.req_id] = pending.cache
        self._req_pool_idx[pending.req_id] = pending.req_pool_idx
        self._req_synced_offset[pending.req_id] = pending.synced_offset
        self._store_auxiliary_state(pending.req_pool_idx, pending.cache)
        return next_token

    def extend_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> MlxPendingExtend:
        """Queue chunked-prefill continuation without evaluating."""
        assert (
            req_id in self._req_caches
        ), f"extend_start called for unknown request {req_id}"

        cache = self._req_caches[req_id]

        input_ids = mx.array([new_token_ids], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)
        lazy_token = mx.argmax(logits[:, -1, :], axis=-1)

        if not self.disable_radix_cache and new_slot_ids:
            synced = self._req_synced_offset[req_id]
            self._sync_new_kv_to_pool(cache, synced, new_slot_ids)
            new_synced_offset = synced + len(new_slot_ids)
        else:
            new_synced_offset = self._req_synced_offset.get(req_id, 0)

        return MlxPendingExtend(
            lazy_token=lazy_token,
            req_id=req_id,
            new_token_ids=list(new_token_ids),
            new_synced_offset=new_synced_offset,
        )

    def extend_finalize(self, pending: MlxPendingExtend) -> int:
        """Materialise a pending extend and commit per-request state."""
        next_token = int(pending.lazy_token.item())

        prev_tokens = self._req_token_ids[pending.req_id]
        if prev_tokens:
            prev_tokens.pop()  # remove stale intermediate token
        prev_tokens.extend(pending.new_token_ids)
        prev_tokens.append(next_token)

        self._req_synced_offset[pending.req_id] = pending.new_synced_offset
        self._store_auxiliary_state(
            self._req_pool_idx[pending.req_id],
            self._req_caches[pending.req_id],
        )
        return next_token

    def _extract_model_components(self):
        """Cache embedding, norm, and lm_head for layer-by-layer hybrid forward."""
        root = getattr(self.model, "language_model", self.model)
        text_model = getattr(root, "model", root)
        embed = text_model.embed_tokens
        norm = text_model.norm
        if hasattr(root, "lm_head"):
            lm_head = root.lm_head
        elif hasattr(root, "args") and getattr(root.args, "tie_word_embeddings", False):
            lm_head = text_model.embed_tokens.as_linear
        else:
            lm_head = root.lm_head
        return embed, norm, lm_head

    def _decode_with_hybrid_batching(
        self,
        caches: list[list[Any]],
        batched_input: mx.array,
        req_ids: list[str],
    ) -> mx.array:
        """Layer-by-layer hybrid decode for attention plus auxiliary state.

        Attention layers run with batched hidden states via
        ``BatchedDecodeContext``. Auxiliary layers run batched when their
        native cache implements mlx-lm's merge/extract protocol, otherwise
        they fall back to per-request execution.
        """
        batch_size = len(caches)

        hidden_states = self._model_embed(batched_input)

        ctx = self._build_batched_decode_context(caches, req_ids)
        seq_lens = ctx.seq_lens
        max_offset = max(seq_lens)

        set_context(ctx)
        try:
            for layer_idx in range(self._cache_layout.num_layers):
                layer = self._cache_layout.layers[layer_idx]

                if self._cache_layout.attention_attrs[layer_idx] is not None:
                    shim = AttentionOffsetCache(offset=max_offset)
                    hidden_states = layer(hidden_states, mask=None, cache=shim)
                else:
                    layer_caches = [caches[i][layer_idx] for i in range(batch_size)]
                    hidden_states = self._decode_auxiliary_layer(
                        layer,
                        hidden_states,
                        layer_caches,
                    )
        finally:
            clear_context()

        hidden_states = self._model_norm(hidden_states)
        logits = self._extract_logits(self._model_lm_head(hidden_states))
        return mx.argmax(logits[:, -1, :], axis=-1)

    def _decode_auxiliary_layer(
        self,
        layer: Any,
        hidden_states: mx.array,
        layer_caches: list[Any],
    ) -> mx.array:
        """Decode one auxiliary layer, batching when native cache supports it."""
        if self._can_batch_auxiliary_layer(layer, layer_caches):
            return self._decode_auxiliary_layer_batched(
                layer,
                hidden_states,
                layer_caches,
            )

        results = []
        for i, cache in enumerate(layer_caches):
            results.append(layer(hidden_states[i : i + 1], mask=None, cache=cache))
        return mx.concatenate(results, axis=0)

    @staticmethod
    def _can_batch_auxiliary_layer(layer: Any, layer_caches: list[Any]) -> bool:
        """Return whether an auxiliary layer can run with merged cache state.

        Qwen3.5/Qwen3-Next DeltaNet layers use the mlx-lm DecoderLayer shape
        below with ``ArraysCache``. Its ``merge``/``extract`` helpers can batch
        native state temporarily and split it back to per-request cache objects.
        """
        if not layer_caches:
            return False
        if not (
            getattr(layer, "is_linear", False)
            and hasattr(layer, "input_layernorm")
            and hasattr(layer, "linear_attn")
            and hasattr(layer, "post_attention_layernorm")
            and hasattr(layer, "mlp")
        ):
            return False

        cache_type = type(layer_caches[0])
        if not callable(getattr(cache_type, "merge", None)) or not all(
            isinstance(cache, cache_type) and callable(getattr(cache, "extract", None))
            for cache in layer_caches
        ):
            return False
        return True

    @staticmethod
    def _decode_auxiliary_layer_batched(
        layer: Any,
        hidden_states: mx.array,
        layer_caches: list[Any],
    ) -> mx.array:
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        batched_cache = MlxModelRunner._merge_auxiliary_caches(layer_caches)
        mixed = layer.linear_attn(normed, mask=None, cache=batched_cache)

        extract = getattr(batched_cache, "extract", None)
        if not callable(extract):
            raise RuntimeError(
                f"{type(batched_cache).__name__}.merge() returned a cache "
                "without extract(); cannot split auxiliary decode state"
            )
        for i, cache in enumerate(layer_caches):
            split_cache = extract(i)
            MlxModelRunner._replace_cache_contents(cache, split_cache)

        hidden_states = residual + mixed
        return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

    @staticmethod
    def _merge_auxiliary_caches(layer_caches: list[Any]) -> Any:
        if MlxModelRunner._can_fast_merge_arrays_cache(layer_caches):
            return MlxModelRunner._fast_merge_arrays_cache(layer_caches)
        return type(layer_caches[0]).merge(layer_caches)

    @staticmethod
    def _can_fast_merge_arrays_cache(layer_caches: list[Any]) -> bool:
        cache_type = type(layer_caches[0])
        if cache_type.__name__ != "ArraysCache":
            return False
        return all(
            type(cache) is cache_type
            and isinstance(getattr(cache, "cache", None), list)
            and getattr(cache, "lengths", None) is None
            and getattr(cache, "left_padding", None) is None
            for cache in layer_caches
        )

    @staticmethod
    def _fast_merge_arrays_cache(layer_caches: list[Any]) -> Any:
        """Merge mlx-lm ArraysCache with concat instead of zero+slice writes."""
        cache_type = type(layer_caches[0])
        merged = cache_type(len(layer_caches[0].cache))
        slots = []
        for slot_idx in range(len(layer_caches[0].cache)):
            values = [cache.cache[slot_idx] for cache in layer_caches]
            first = next((value for value in values if value is not None), None)
            if first is None:
                slots.append(None)
                continue
            slots.append(
                mx.concatenate(
                    [
                        value if value is not None else mx.zeros_like(first)
                        for value in values
                    ],
                    axis=0,
                )
            )
        merged.cache = slots
        return merged

    @staticmethod
    def _replace_cache_contents(cache: Any, new_cache: Any) -> None:
        """Replace cache contents while preserving the original cache object."""
        if type(cache) is type(new_cache) and hasattr(cache, "__dict__"):
            cache.__dict__.clear()
            cache.__dict__.update(new_cache.__dict__)
            return
        if hasattr(cache, "state") and hasattr(new_cache, "state"):
            cache.state = new_cache.state
            return
        raise RuntimeError(
            f"Cannot copy {type(new_cache).__name__} state into "
            f"{type(cache).__name__}"
        )

    def _decode_with_native_cache(
        self,
        caches: list[list[Any]],
        input_ids_by_request: list[mx.array],
    ) -> mx.array:
        lazy_token_list = []
        for input_ids, cache in zip(input_ids_by_request, caches):
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            lazy_token_list.append(mx.argmax(logits[:, -1, :], axis=-1))
        return (
            lazy_token_list[0]
            if len(lazy_token_list) == 1
            else mx.concatenate(lazy_token_list, axis=0)
        )

    def _decode_with_batched_attention(
        self,
        caches: list[list[Any]],
        batched_input: mx.array,
        req_ids: list[str],
    ) -> mx.array:
        ctx = self._build_batched_decode_context(caches, req_ids)
        seq_lens = ctx.seq_lens
        set_context(ctx)
        try:
            max_offset = max(seq_lens)
            shim_cache = [
                AttentionOffsetCache(offset=max_offset)
                for _ in range(self._cache_layout.num_layers)
            ]
            model_output = self.model(batched_input, cache=shim_cache)
            logits = self._extract_logits(model_output)
            return mx.argmax(logits[:, -1, :], axis=-1)
        finally:
            clear_context()

    def _build_batched_decode_context(
        self,
        caches: list[list[Any]],
        req_ids: list[str],
    ) -> BatchedDecodeContext:
        """Build the shared attention/AOT context for one decode step."""
        return BatchedDecodeContext.from_decode(
            caches=caches,
            req_ids=req_ids,
            aot_kernels=self._aot_kernels,
            kv_pool=self._attention_kv_pool,
            req_pool_idx=self._req_pool_idx,
            req_to_token_pool=self._req_to_token_pool,
            attention_layer_indices=self._cache_layout.attention_layer_indices,
            attention_pool_index_by_layer=(
                self._cache_layout.attention_pool_index_by_layer
            ),
        )

    def decode_batch_start(self, req_ids: list[str]) -> MlxPendingDecode:
        """Queue a decode forward pass without evaluating.

        The caller is responsible for calling ``mx.async_eval`` on the
        returned ``lazy_tokens`` (and optionally per-cache state arrays)
        to kick off GPU work before :meth:`decode_batch_finalize`.
        """
        caches = [self._req_caches[rid] for rid in req_ids]
        last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        if self._cache_layout.has_auxiliary_state:
            lazy_tokens = self._decode_with_hybrid_batching(
                caches, batched_input, list(req_ids)
            )
        else:
            lazy_tokens = self._decode_with_batched_attention(
                caches, batched_input, list(req_ids)
            )

        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=list(req_ids),
            caches=caches,
        )

    def decode_batch_start_chained(
        self,
        prev: MlxPendingDecode,
    ) -> MlxPendingDecode:
        """Build the next decode step on top of a still-lazy previous decode.

        Feeds ``prev.lazy_tokens`` (an unevaluated ``mx.array`` of shape
        ``(B,)``) as the next step's input ids, reusing
        ``prev.caches`` in-place so that per-layer attention KV writes from
        step N and step N+1 land in the same buffers.  MLX
        tracks the full dependency graph, so once ``mx.async_eval`` is
        called the GPU executes N+1 immediately after N with no gap.

        Caller contract:

        * ``prev`` MUST refer to the same set of requests (same order) as
          the batch the caller intends to run next.  Composition changes
          (finished reqs, new prefills) must break the chain instead.
        * After calling this, finalise ``prev`` BEFORE finalising the
          returned pending: state bookkeeping for step N has to happen
          before step N+1's bookkeeping.
        """
        caches = prev.caches

        # After prev's graph ran, each attention KV cache offset was
        # bumped by one per layer - attention wrapper's `write_token`
        # mutates the Python offset synchronously at graph-build time.
        # So layer-0 offsets reflect the position the NEW token will
        # be written at in step N+1 (and equivalently the RoPE offset).
        batched_input = prev.lazy_tokens[:, None]
        if self._cache_layout.has_auxiliary_state:
            lazy_tokens = self._decode_with_hybrid_batching(
                caches, batched_input, prev.req_ids
            )
        else:
            lazy_tokens = self._decode_with_batched_attention(
                caches, batched_input, prev.req_ids
            )

        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=caches,
        )

    def decode_batch_finalize(
        self,
        pending: MlxPendingDecode,
    ) -> list[int]:
        """Materialise a pending decode and update per-request token lists.

        ``pending.lazy_tokens.tolist()`` implicitly blocks until that
        specific lazy array (and its graph ancestors, including the
        per-request cache writes for this step) is evaluated.  The
        caller should have previously handed this pending's lazy_tokens
        to ``mx.async_eval`` (or to a subsequent chained step that will
        be async_eval'd).
        """
        raw = pending.lazy_tokens.tolist()
        if not isinstance(raw, list):
            raw = [raw]
        next_tokens = [int(t) for t in raw]

        for i, rid in enumerate(pending.req_ids):
            self._req_token_ids[rid].append(next_tokens[i])

        self._decode_step_ct += 1
        if self._clear_steps > 0 and self._decode_step_ct % self._clear_steps == 0:
            mx.clear_cache()

        return next_tokens

    def has_request(self, req_id: str) -> bool:
        """Check if a request has active state."""
        return req_id in self._req_caches

    def remove_request(self, req_id: str):
        """Sync remaining decode KV to pool, then release request state."""
        if not self.disable_radix_cache:
            self._sync_decode_kv_to_pool(req_id)

        self._req_token_ids.pop(req_id, None)
        cache = self._req_caches.pop(req_id, None)
        if cache is not None:
            self._release_cache(cache)
        self._req_pool_idx.pop(req_id, None)
        self._req_synced_offset.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._req_token_ids.clear()
        for cache in self._req_caches.values():
            self._release_cache(cache)
        self._req_caches.clear()
        self._req_pool_idx.clear()
        self._req_synced_offset.clear()
        if self._attention_kv_pool is not None:
            self._attention_kv_pool.clear()
