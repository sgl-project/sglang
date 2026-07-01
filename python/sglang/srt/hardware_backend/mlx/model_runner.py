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
    get_num_kv_heads,
    patch_model_attention,
    set_context,
    uses_sliding_window_attention,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import get_global_server_args

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
    pool_synced_offsets: dict[str, int] | None = None


_MLX_QUANTIZATION_PRESETS: dict[str, tuple[int, int]] = {
    # name -> (bits, group_size). group_size=64 matches the mlx-community convention.
    "mlx_q4": (4, 64),
    "mlx_q8": (8, 64),
}
_MLX_KV_FLOAT_DTYPES = {mx.float16, mx.bfloat16, mx.float32}


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
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.disable_radix_cache = disable_radix_cache
        self._mem_fraction_static = mem_fraction_static
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

        chunk_size = get_global_server_args().mamba_cache_chunk_size
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
            # QuantizedLinear stores packed weights as integers, while the KV
            # cache stores dequantized projection outputs.
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
        """Determine pool slot count (auto-size from available memory if needed)."""
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
        pool_size = max(kv_budget // bytes_per_slot, 256)
        logger.info(
            f"Auto-sized attention KV pool: "
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
    ) -> tuple[mx.array, dict[str, int] | None]:
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
            pool_synced_offsets = None
            if ctx.aot.rope is not None and ctx.aot.rope.new_token_slots is not None:
                pool_synced_offsets = {
                    rid: self._first_attention_cache(cache).offset
                    for rid, cache in zip(req_ids, caches)
                }
            return mx.argmax(logits[:, -1, :], axis=-1), pool_synced_offsets
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
            paged_attention_supported=not self._cache_layout.has_auxiliary_state,
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
            pool_synced_offsets = None
        else:
            lazy_tokens, pool_synced_offsets = self._decode_with_batched_attention(
                caches, batched_input, list(req_ids)
            )

        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=list(req_ids),
            caches=caches,
            pool_synced_offsets=pool_synced_offsets,
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

        # TODO (changminbark): Need to fix
        # ContiguousAttentionKVCache.write_token to accommodate dynamic growing
        # like ContiguousAttentionKVCache.update_and_fetch.

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
            pool_synced_offsets = None
        else:
            lazy_tokens, pool_synced_offsets = self._decode_with_batched_attention(
                caches, batched_input, prev.req_ids
            )

        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=caches,
            pool_synced_offsets=pool_synced_offsets,
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
            if pending.pool_synced_offsets is not None:
                self._req_synced_offset[rid] = pending.pool_synced_offsets[rid]

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
