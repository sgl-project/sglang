"""Lightweight ModelRunner stub for MLX on Apple Silicon.

Skips PyTorch weight loading.  Creates only the CPU-side bookkeeping
(req_to_token_pool, token_to_kv_pool_allocator) the scheduler needs.
"""

import logging
from typing import Tuple

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class _DummyKVCache(KVCache):
    """A KV cache that allocates no GPU memory.

    Satisfies the KVCache interface so that TokenToKVPoolAllocator can be
    constructed, but every buffer access raises — the MLX backend manages
    its own KV cache internally.
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
        raise RuntimeError("_DummyKVCache has no key buffer (MLX manages KV cache)")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError("_DummyKVCache has no value buffer (MLX manages KV cache)")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("_DummyKVCache has no kv buffer (MLX manages KV cache)")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        raise RuntimeError("_DummyKVCache cannot set kv buffer (MLX manages KV cache)")

    def get_kv_size_bytes(self):
        return 0, 0


class _DummyMambaPool:
    """Bookkeeping-only mamba pool for MLX hybrid models.

    MLX keeps linear-attention state in native mlx-lm caches.  The scheduler
    still needs mamba pool indices for radix bookkeeping, so this pool tracks
    slots without allocating state tensors.
    """

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self.mamba_cache = None
        self.mem_usage = 0
        self.clear()

    def available_size(self):
        return int(self.free_slots.numel())

    def alloc(self, need_size: int):
        if need_size > self.available_size():
            return None
        slots = self.free_slots[:need_size].clone()
        self.free_slots = self.free_slots[need_size:]
        return slots

    def free(self, indices):
        if indices is None:
            return
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.device).view(
            -1
        )
        if indices.numel() == 0:
            return
        self.free_slots = torch.cat([self.free_slots, indices])

    def fork_from(self, src):
        dst = self.alloc(1)
        if dst is None:
            return None
        self.copy_from(src, dst)
        return dst

    def copy_from(self, src, dst):
        return None

    def clear(self):
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )


class _DummyMambaReqToTokenPool(ReqToTokenPool):
    """Req-to-token pool with mamba slot bookkeeping but no mamba tensors."""

    def __init__(
        self,
        *,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        mamba_size: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        self.mamba_pool = _DummyMambaPool(size=mamba_size, device=device)
        self.enable_mamba_extra_buffer = False
        self.req_index_to_mamba_index_mapping = torch.zeros(
            self._alloc_size, dtype=torch.int32, device=device
        )

    def alloc(self, reqs):
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices = []
        for req in reqs:
            if getattr(req, "mamba_pool_idx", None) is not None:
                mid = req.mamba_pool_idx
            else:
                allocated = self.mamba_pool.alloc(1)
                assert allocated is not None, "Not enough dummy mamba slots"
                mid = allocated[0]
                req.mamba_pool_idx = mid
            mamba_indices.append(mid.to(dtype=torch.int32))
        self.req_index_to_mamba_index_mapping[select_index] = torch.stack(mamba_indices)
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        return 0

    def free_mamba_cache(self, req, mamba_ping_pong_track_buffer_to_keep=None):
        if getattr(req, "mamba_pool_idx", None) is not None:
            self.mamba_pool.free(req.mamba_pool_idx.unsqueeze(0))
            req.mamba_pool_idx = None

    def free(self, req):
        super().free(req)

    def clear(self):
        super().clear()
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()


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

    def initialize(self, pre_model_load_memory: float):
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
        self.max_running_requests = min(
            self.max_total_num_tokens // 2,
            4096,
        )
        self.is_hybrid_swa = False

        # Create minimal pools
        if self.mambaish_config is not None:
            mamba_size = self.server_args.max_mamba_cache_size
            if mamba_size is None:
                mamba_size = self.max_running_requests * 4
            self.req_to_token_pool = _DummyMambaReqToTokenPool(
                size=self.max_running_requests,
                max_context_len=self.model_config.context_len,
                device="cpu",
                enable_memory_saver=False,
                mamba_size=mamba_size,
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
        self.graph_runner = None
        self.graph_mem_usage = 0
        self.attn_backend = None

        logger.info(
            f"MLX stub: initialized minimal pools "
            f"(max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"zero GPU KV cache allocation)"
        )
