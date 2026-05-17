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
