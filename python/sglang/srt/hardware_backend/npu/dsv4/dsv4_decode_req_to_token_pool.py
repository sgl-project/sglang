"""DecodeReqToTokenPool + the five DSV4 per-req tables (swa/c4/c128/c4_state/
c128_state) the NPU attention and KV-transfer paths need on the decode side."""

from __future__ import annotations

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class DSV4NPUDecodeReqToTokenPool(DecodeReqToTokenPool):
    """DecodeReqToTokenPool + DSV4 swa/c4/c128(+state) per-req tables;
    see :class:`DSV4NPUReqToTokenPool` for the table semantics."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        # Allocator back-ref (wired via register_dsv4_allocator) so free(req)
        # releases c4/c128 pages; None at construction for base clear().
        self._dsv4_allocator = None

        # (name, columns): swa/state = 1 slot/token, c4/c128 = 1 slot/ratio tokens.
        # Init zero so unallocated columns map to block 0 (kernel skip sentinel).
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            for name, cols in (
                ("req_to_token_swa", max_context_len),
                ("req_to_token_c4", max(1, max_context_len // 4)),
                ("req_to_token_c128", max(1, max_context_len // 128)),
                ("req_to_token_c4_state", max_context_len),
                ("req_to_token_c128_state", max_context_len),
            ):
                setattr(
                    self,
                    name,
                    torch.zeros(
                        (self._alloc_size, cols),
                        dtype=torch.int32,
                        device=device,
                    ),
                )

    # Per-pool write helpers, called by mem_cache/common.py after alloc with
    # slot indices from DSV4OutCacheLoc.
    def write_swa(self, indices, values: torch.Tensor) -> None:
        self.req_to_token_swa[indices] = values

    def write_c4(self, indices, values: torch.Tensor) -> None:
        self.req_to_token_c4[indices] = values

    def write_c128(self, indices, values: torch.Tensor) -> None:
        self.req_to_token_c128[indices] = values

    def write_c4_state(self, indices, values: torch.Tensor) -> None:
        self.req_to_token_c4_state[indices] = values

    def write_c128_state(self, indices, values: torch.Tensor) -> None:
        self.req_to_token_c128_state[indices] = values

    def register_dsv4_allocator(self, allocator) -> None:
        """Wire the DSV4NPUTokenToKVPoolAllocator ref so ``free(req)`` can
        release c4/c128 pool pages alongside the req_pool_idx slot."""
        self._dsv4_allocator = allocator

    def free(self, req):
        # Release c4/c128 via the allocator (None before register_dsv4_allocator).
        if self._dsv4_allocator is not None:
            self._dsv4_allocator.free(req=req, req_to_token_pool=self)
        super().free(req)
