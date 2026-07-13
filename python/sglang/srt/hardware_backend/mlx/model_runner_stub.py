"""Lightweight ModelRunner stub for MLX on Apple Silicon.

Skips PyTorch weight loading.  Creates only the CPU-side bookkeeping
(req_to_token_pool, token_to_kv_pool_allocator) the scheduler needs.
"""

import logging
from typing import Tuple

import torch

from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
    MlxAuxiliaryStateReqToTokenPool,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Ratio of auxiliary-state slots to concurrently running requests on hybrid /
# linear-attention models. Each running request holds one live slot
# (MlxAuxiliaryStateReqToTokenPool.alloc), and the headroom covers radix-held
# snapshots. Used for BOTH the default pool sizing and the concurrency bound
# in _resolve_max_running_requests so the two cannot drift apart.
MLX_AUX_STATE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 4


class _DummyKVCache(KVCache):
    """Scheduler-facing KV cache that allocates no GPU memory.

    Satisfies the KVCache interface so that TokenToKVPoolAllocator can be
    constructed, but every buffer access raises. The MLX backend manages
    attention KV and auxiliary state internally.
    """

    def __init__(self, size: int, dtype: torch.dtype, device: str):
        # Bypass KVCache.__init__ to avoid custom_mem_pool / memory_saver
        # initialization that may touch CUDA APIs.
        self.size = size
        self.page_size = 1
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError("_DummyKVCache has no key buffer (MLX manages cache)")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError("_DummyKVCache has no value buffer (MLX manages cache)")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("_DummyKVCache has no kv buffer (MLX manages cache)")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        raise RuntimeError("_DummyKVCache cannot set kv buffer (MLX manages cache)")

    def get_kv_size_bytes(self):
        return 0, 0


class _DummyModel:
    """Minimal stand-in so that `inspect.signature(model.forward)` and
    `getattr(model, ...)` calls in ModelRunner.__init__ don't crash."""

    @staticmethod
    def forward():
        pass


class MlxModelRunnerStub(ModelRunner):
    """ModelRunner that skips PyTorch weight loading and KV cache allocation.

    Overrides both load_model() and initialize() so that no PyTorch model
    weights are loaded and no large KV cache tensors are allocated.  Only
    the minimal bookkeeping pools needed by the scheduler are created.
    """

    # No KV canary on the MLX path. The base ModelRunner installs it via
    # install_canary() in its full initialize(), which this lightweight override
    # skips. Downstream consumers (scheduler, cuda graph runner, speculative
    # workers) all guard with `canary_manager is not None`, so default to None
    # as a class attribute to keep those checks working instead of raising
    # AttributeError.
    canary_manager = None

    # No prefill-aware SWA on the MLX path. The base ModelRunner derives this in
    # its full initialize() from `model.is_prefill_aware_swa()`, which this
    # lightweight override skips (and `_DummyModel` does not implement). The
    # scheduler reads `model_runner.prefill_aware_swa` unconditionally when
    # admitting a prefill batch, so default to False as a class attribute to keep
    # that path working instead of raising AttributeError.
    prefill_aware_swa = False

    def __init__(self, *args, mlx_pool_size: int | None = None, **kwargs):
        self._mlx_pool_size = mlx_pool_size
        super().__init__(*args, **kwargs)

    def load_model(self):
        """Set only the metadata that downstream code needs, without
        loading any PyTorch model weights."""
        logger.info(
            "MLX stub: skipping PyTorch model weight loading "
            "(inference runs through MLX)"
        )

        self.model = _DummyModel()

        self.sliding_window_size = None
        if (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size

        self.dtype = self.model_config.dtype
        self.weight_load_mem_usage = 0

    def _resolve_max_running_requests(self) -> int:
        """Concurrency cap handed to the scheduler.

        Honors ``--max-running-requests``, mirroring the base runner's clamp
        (``model_runner_kv_cache_mixin._resolve_max_num_reqs``): the requested
        value is split per dp worker and capped by the KV pool capacity. When
        the flag is unset, fall back to a capacity-based default.

        On hybrid / linear-attention models the concurrency is additionally
        bounded by the auxiliary-state pool: each running request allocates one
        slot out of ``max_mamba_cache_size`` (asserting when exhausted), so a
        cap the pool cannot back would crash mid-serving instead of failing
        at startup. Mirrors the base resolver's mamba bound and zero-reject.

        Requires ``self.max_total_num_tokens`` to already be set.
        """
        capacity_cap = self.max_total_num_tokens // 2
        requested = self.server_args.max_running_requests
        if requested is None:
            requested_per_worker = None
            resolved = min(capacity_cap, 4096)
        else:
            requested_per_worker = requested // self.dp_size
            resolved = min(requested_per_worker, capacity_cap)

        aux_state_size = self.server_args.max_mamba_cache_size
        if self.mambaish_config is not None and aux_state_size is not None:
            ratio = MLX_AUX_STATE_SIZE_MAX_RUNNING_REQUESTS_RATIO
            resolved = min(resolved, aux_state_size // ratio)
            if resolved <= 0:
                raise RuntimeError(
                    f"MLX auxiliary-state cache is too small to serve any "
                    f"requests: max_mamba_cache_size={aux_state_size} backs "
                    f"only {aux_state_size // ratio} concurrent requests "
                    f"({ratio} slots per request). Increase "
                    f"--max-mamba-cache-size to at least {ratio}, or leave it "
                    f"unset to size the pool from the concurrency cap."
                )

        if requested_per_worker is not None and resolved < requested_per_worker:
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per dp worker) due to the available KV cache or "
                "auxiliary-state capacity.",
                requested_per_worker,
                resolved,
            )
        return resolved

    def initialize(self):
        """Lightweight initialize that skips heavy PyTorch setup.

        Creates minimal req_to_token_pool and token_to_kv_pool_allocator
        with a dummy KV cache (zero GPU memory) so the scheduler works.
        """
        from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        # Load model (sets metadata only)
        self.sampler = None
        self.load_model()

        # Layer metadata
        model_num_layers = max(
            self.model_config.num_hidden_layers,
            self.model_config.num_attention_layers,
        )
        self.start_layer = 0
        self.end_layer = model_num_layers
        self.num_effective_layers = model_num_layers

        # KV cache dtype
        self.kv_cache_dtype = self.dtype

        # Pool sizing — use the MLX runner's auto-sized pool if available,
        # otherwise fall back to context_len.
        if self._mlx_pool_size is not None:
            self.max_total_num_tokens = self._mlx_pool_size
        else:
            self.max_total_num_tokens = self.model_config.context_len
        self.max_running_requests = self._resolve_max_running_requests()
        self.is_hybrid_swa = False

        # Create minimal pools
        if self.mambaish_config is not None:
            auxiliary_state_size = self.server_args.max_mamba_cache_size
            if auxiliary_state_size is None:
                auxiliary_state_size = (
                    self.max_running_requests
                    * MLX_AUX_STATE_SIZE_MAX_RUNNING_REQUESTS_RATIO
                )
            self.req_to_token_pool = MlxAuxiliaryStateReqToTokenPool(
                size=self.max_running_requests,
                max_context_len=self.model_config.context_len,
                device="cpu",
                enable_memory_saver=False,
                auxiliary_state_size=auxiliary_state_size,
            )
        else:
            self.req_to_token_pool = ReqToTokenPool(
                size=self.max_running_requests,
                max_context_len=self.model_config.context_len,
                device="cpu",
                enable_memory_saver=False,
            )

        dummy_kv = _DummyKVCache(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
        )
        self.token_to_kv_pool = dummy_kv
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
            kvcache=dummy_kv,
            need_sort=False,
        )

        # No CUDA graphs, no attention backend
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0
        self.attn_backend = None

        logger.info(
            f"MLX stub: initialized minimal pools "
            f"(max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"zero GPU KV cache allocation)"
        )

    def alloc_memory_pool(self, memory_pool_config=None):
        """No-op: MLX manages its own KV cache."""
        pass
