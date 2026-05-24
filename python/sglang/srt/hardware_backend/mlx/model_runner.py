"""MLX model runner for Apple Silicon.

Slot allocation and radix-trie prefix matching are handled by the
scheduler (``TokenToKVPoolAllocator`` / ``RadixCache``).  This runner
reads cached KV from ``MlxKVPool``, runs the forward pass, and writes
new KV back.  Each request also keeps a ``ContiguousKVCache`` for
decode-time attention.

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

import mlx.core as mx
import psutil
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_lm_load
from mlx_lm.utils import quantize_model as mlx_lm_quantize_model

from sglang.srt.hardware_backend.mlx.kv_cache import (
    BatchedDecodeContext,
    ContiguousKVCache,
    MLXAttentionWrapper,
    OffsetCache,
    PoolBackedCache,
    clear_context,
    find_attention_layers,
    get_num_layers,
    patch_model_attention,
    set_context,
)
from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


@dataclass
class MlxPendingPrefill:
    """Lazy prefill state, finalised after ``mx.eval``/``async_eval``.

    ``cache`` is the per-layer list of ``ContiguousKVCache`` that will
    become ``_req_caches[req_id]`` once the request is committed.  It
    may have been converted from a transient ``PoolBackedCache`` list
    already (so its ``state`` arrays are safe to hand to ``async_eval``).
    """

    lazy_token: mx.array
    cache: list  # list[ContiguousKVCache]
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

    ``caches`` is a per-request list of per-layer ``ContiguousKVCache``
    references (``caches[req_idx][layer_idx]``).  These are the same
    objects the attention wrapper writes into during the forward pass,
    so :meth:`decode_batch_start_chained` can launch the next step on
    top of the same caches without materialising this step first.
    """

    lazy_tokens: mx.array
    req_ids: list[str]
    caches: list  # list[list[ContiguousKVCache]]


_MLX_QUANTIZATION_PRESETS: dict[str, tuple[int, int]] = {
    # name -> (bits, group_size). group_size=64 matches the mlx-community convention.
    "mlx_q4": (4, 64),
    "mlx_q8": (8, 64),
}


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
        # On-the-fly quantization preset (e.g. "mlx_q4"). None = no on-load quantization.
        # Pre-quantized HF repos (e.g. mlx-community/Qwen3-0.6B-4bit) load correctly
        # regardless of this setting — mlx_lm.load() detects the config and instantiates
        # QuantizedLinear modules directly.
        self._quantization: str | None = quantization

        self._load_model()

        # Pin MLX allocations to prevent OS paging
        device_info = mx.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            mx.set_wired_limit(max_wired)
            logger.info(f"Wired memory limit set to {max_wired / (1024**3):.1f} GB")

        patch_model_attention(self.model)

        self._num_layers = get_num_layers(self.model)
        self._max_seq_len = 4096  # doubles on overflow

        self._req_caches: dict[str, list[ContiguousKVCache | PoolBackedCache]] = {}
        self._req_token_ids: dict[str, list[int]] = {}
        self._cache_pool: list[list[ContiguousKVCache]] = []  # reusable caches

        self._kv_pool: MlxKVPool | None = None
        self._req_to_token_pool: ReqToTokenPool | None = None
        self._req_pool_idx: dict[str, int] = {}
        self._req_synced_offset: dict[str, int] = {}

        self._pool_size = self._compute_pool_size(pool_size)

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _acquire_cache(self) -> list[ContiguousKVCache]:
        """Get a reusable cache list from the pool, or create a new one."""
        if self._cache_pool:
            cache = self._cache_pool.pop()
            for c in cache:
                c.offset = 0
            return cache
        return [
            ContiguousKVCache(max_seq_len=self._max_seq_len)
            for _ in range(self._num_layers)
        ]

    def _release_cache(self, cache: list[ContiguousKVCache]) -> None:
        """Return a cache list to the pool for reuse."""
        self._cache_pool.append(cache)

    @staticmethod
    def _eval_with_cache(
        token_result: mx.array, cache: list[ContiguousKVCache | PoolBackedCache]
    ) -> None:
        """Evaluate token result and all cache buffers in one mx.eval call."""
        mx.eval(token_result, *[s for c in cache for s in c.state])

    @staticmethod
    def _cache_state_arrays(
        pending_caches: list[list[ContiguousKVCache | PoolBackedCache]],
    ) -> list[mx.array]:
        """Flatten pending decode cache state list into an array list.

        Safe to hand to ``mx.async_eval``.
        """
        return [
            s
            for cache_list in pending_caches
            for cache in cache_list
            for s in cache.state
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
        # actual usage before KV pool sizing.
        mx.eval(self.model.parameters())

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def _get_attn_config(self) -> tuple[int, int, mx.Dtype]:
        """Return (n_kv_heads, head_dim, dtype) from the model."""
        layer_list, attn_attr = find_attention_layers(self.model)
        if not layer_list:
            raise RuntimeError("Cannot determine attention config: no layers found")
        sample_attn = getattr(layer_list[0], attn_attr)
        if isinstance(sample_attn, MLXAttentionWrapper):
            sample_attn = sample_attn._inner
        n_kv_heads = sample_attn.n_kv_heads
        if hasattr(sample_attn, "head_dim"):
            head_dim = sample_attn.head_dim
        elif hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            head_dim = sample_attn.k_proj.weight.shape[0] // n_kv_heads
        else:
            raise RuntimeError("Cannot determine head_dim from attention module")
        dtype = mx.float16
        if hasattr(sample_attn, "k_proj") and hasattr(sample_attn.k_proj, "weight"):
            dtype = sample_attn.k_proj.weight.dtype
        return n_kv_heads, head_dim, dtype

    def _compute_pool_size(self, explicit_size: int | None) -> int:
        """Determine pool slot count (auto-size from available memory if needed)."""
        if explicit_size is not None:
            return explicit_size
        n_kv_heads, head_dim, dtype = self._get_attn_config()
        num_layers = self._num_layers
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
            f"Auto-sized KV pool: "
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

    def init_kv_pool(self, req_to_token_pool: ReqToTokenPool) -> None:
        """Create MlxKVPool (+1 for padding slot 0) and wire scheduler pools."""
        self._req_to_token_pool = req_to_token_pool
        if self.disable_radix_cache:
            return
        n_kv_heads, head_dim, dtype = self._get_attn_config()
        # +1 for padding slot 0
        self._kv_pool = MlxKVPool(
            pool_size=self._pool_size + 1,
            num_layers=self._num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        logger.info(
            f"KV pool initialized: pool_size={self._pool_size} "
            f"(buffer size {self._pool_size + 1} incl. padding slot 0), "
            f"{self._num_layers} layers, {n_kv_heads} kv_heads, {head_dim} head_dim"
        )

    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
    ) -> int:
        """Prefill a request.  Returns next_token_id."""
        pending = self.prefill_start(
            req_id=req_id,
            new_token_ids=new_token_ids,
            full_token_ids=full_token_ids,
            prefix_slot_ids=prefix_slot_ids,
            new_slot_ids=new_slot_ids,
            req_pool_idx=req_pool_idx,
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
        cache: list[ContiguousKVCache],
        cache_start: int,
        slot_ids: list[int],
    ) -> None:
        """Sync KV from contiguous cache to pool at the given slot IDs."""
        if not slot_ids or self._kv_pool is None:
            return
        num_layers = len(cache)
        end = cache_start + len(slot_ids)
        slot_ids_mx = mx.array(slot_ids, dtype=mx.int32)
        # TODO: Standardize ContiguousKVCache size to avoid transpose
        # Transpose cache (1, n_kv_heads, S, head_dim) → pool (S, n_kv_heads, head_dim)
        k_all = mx.stack(
            [
                cache[i].keys[0, :, cache_start:end, :].transpose(1, 0, 2)
                for i in range(num_layers)
            ]
        )
        v_all = mx.stack(
            [
                cache[i].values[0, :, cache_start:end, :].transpose(1, 0, 2)
                for i in range(num_layers)
            ]
        )
        self._kv_pool.set_kv_all_layers(slot_ids_mx, k_all, v_all)

    def _sync_decode_kv_to_pool(self, req_id: str) -> None:
        """Sync un-flushed decode KV for *req_id* to the shared pool."""
        if self._kv_pool is None or self._req_to_token_pool is None:
            return
        cache = self._req_caches.get(req_id)
        if cache is None:
            return
        current_offset = cache[0].offset
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
        if self.disable_radix_cache or self._kv_pool is None:
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
    ) -> MlxPendingPrefill:
        """Queue a prefill forward pass without evaluating.

        Returns an :class:`MlxPendingPrefill` containing the lazy
        next-token ``mx.array`` plus everything needed to commit the
        request in :meth:`prefill_finalize`.  The caller drives the GPU
        by handing ``lazy_token`` (and cache state) to ``mx.async_eval``.
        """
        num_layers = self._num_layers
        prefix_len = len(prefix_slot_ids)

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

        assert self._kv_pool is not None

        new_token_count = len(new_token_ids)

        if prefix_len > 0:
            slot_ids_mx = mx.array(prefix_slot_ids, dtype=mx.int32)
            cache = [
                PoolBackedCache(self._kv_pool, i, slot_ids_mx, prefix_len)
                for i in range(num_layers)
            ]
        else:
            cache = self._acquire_cache()

        if new_token_count > 0:
            extend_tokens = new_token_ids
        else:
            # Full cache hit — rerun last token to get next-token logits
            extend_tokens = full_token_ids[-1:]
            for c in cache:
                c.offset = max(c.offset - 1, 0)

        input_ids = mx.array([extend_tokens], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        lazy_token = mx.argmax(last_logits, axis=-1)

        # Convert PoolBackedCache → ContiguousKVCache for decode.
        # This appends a lazy slice-assign onto the forward graph; the
        # arrays get materialised when the caller evaluates lazy_token.
        if prefix_len > 0:
            contiguous_cache = self._acquire_cache()
            for layer_idx in range(num_layers):
                pbc = cache[layer_idx]
                contiguous_cache[layer_idx].update_and_fetch(
                    pbc._full_keys, pbc._full_values
                )
            cache = contiguous_cache

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
        return next_token

    def decode_batch_start(self, req_ids: list[str]) -> MlxPendingDecode:
        """Queue a decode forward pass without evaluating.

        The caller is responsible for calling ``mx.async_eval`` on the
        returned ``lazy_tokens`` (and optionally per-cache state arrays)
        to kick off GPU work before :meth:`decode_batch_finalize`.
        """
        batch_size = len(req_ids)
        num_layers = self._num_layers

        caches = [self._req_caches[rid] for rid in req_ids]

        if batch_size == 1:
            cache = caches[0]
            last_token = self._req_token_ids[req_ids[0]][-1]
            input_ids = mx.array([[last_token]], dtype=mx.int32)
            model_output = self.model(input_ids, cache=cache)
            logits = self._extract_logits(model_output)
            lazy_tokens = mx.argmax(logits[:, -1, :], axis=-1)
            return MlxPendingDecode(
                lazy_tokens=lazy_tokens,
                req_ids=list(req_ids),
                caches=caches,
            )

        seq_lens = [caches[i][0].offset for i in range(batch_size)]
        layer_caches = [
            [caches[i][layer_idx] for i in range(batch_size)]
            for layer_idx in range(num_layers)
        ]
        ctx = BatchedDecodeContext(
            batch_size=batch_size,
            seq_lens=seq_lens,
            layer_caches=layer_caches,
        )
        set_context(ctx)
        try:
            max_offset = max(seq_lens)
            shim_cache = [OffsetCache(offset=max_offset) for _ in range(num_layers)]
            last_tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
            batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
            model_output = self.model(batched_input, cache=shim_cache)
            logits = self._extract_logits(model_output)
            lazy_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        finally:
            clear_context()

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
        ``prev.caches`` in-place so that the per-layer ``ContiguousKVCache``
        writes from step N and step N+1 land in the same buffers.  MLX
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
        batch_size = len(prev.req_ids)
        num_layers = self._num_layers
        caches = prev.caches

        # TODO (changminbark): Need to fix ContiguousKVCache.write_token
        # to accommodate dynamic growing like ContiguousKVCache.update_and_fetch.

        # After prev's graph ran, each ContiguousKVCache.offset was
        # bumped by one per layer — attention wrapper's `write_token`
        # mutates the Python offset synchronously at graph-build time.
        # So layer-0 offsets reflect the position the NEW token will
        # be written at in step N+1 (and equivalently the RoPE offset).
        seq_lens = [caches[i][0].offset for i in range(batch_size)]

        if batch_size == 1:
            cache = caches[0]
            batched_input = prev.lazy_tokens[:, None]
            model_output = self.model(batched_input, cache=cache)
            logits = self._extract_logits(model_output)
            lazy_tokens = mx.argmax(logits[:, -1, :], axis=-1)
            return MlxPendingDecode(
                lazy_tokens=lazy_tokens,
                req_ids=prev.req_ids,
                caches=caches,
            )

        layer_caches = [
            [caches[i][layer_idx] for i in range(batch_size)]
            for layer_idx in range(num_layers)
        ]
        ctx = BatchedDecodeContext(
            batch_size=batch_size,
            seq_lens=seq_lens,
            layer_caches=layer_caches,
        )
        set_context(ctx)
        try:
            max_offset = max(seq_lens)
            shim_cache = [OffsetCache(offset=max_offset) for _ in range(num_layers)]
            batched_input = prev.lazy_tokens[:, None]
            model_output = self.model(batched_input, cache=shim_cache)
            logits = self._extract_logits(model_output)
            lazy_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        finally:
            clear_context()

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
        # TODO (changminbark): allow for flag configuration for clearing mx cache
        if self._decode_step_ct % 256 == 0:
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
            self._cache_pool.append(cache)
        self._req_caches.clear()
        self._req_pool_idx.clear()
        self._req_synced_offset.clear()
        if self._kv_pool is not None:
            self._kv_pool.clear()
