"""DSV4-NPU per-request mapping pool.

Subclass of ``ReqToTokenPool`` that adds five auxiliary per-request tables
needed by the DSV4 attention backend:

  * ``req_to_token_swa``       — slot ids in the SWA full-pool view
  * ``req_to_token_c4``        — slot ids in the c4 compressed-KV pool
  * ``req_to_token_c128``      — slot ids in the c128 compressed-KV pool
  * ``req_to_token_c4_state``  — slot ids in the c4 state-buffer pool
  * ``req_to_token_c128_state``— slot ids in the c128 state-buffer pool

Each table is sized ``[size, max_context_len // ratio]`` for compressed
pools and ``[size, max_context_len]`` for swa / state pools, mirroring the
``req_to_token`` layout. Elements are token-level slot ids; the attention
backend converts to page ids via ``// page_size`` when constructing PA_ND
block tables.

Memory cost example (size=64, max_context_len=32K): swa 8MB + c4 2MB +
c128 64KB + c4_state 8MB + c128_state 8MB ≈ 26MB extra on top of the base
req_to_token (8MB). Negligible vs. the c4/c128 KV pools themselves.

The tables are populated by hooks in ``mem_cache/common.py`` immediately
after a successful alloc_extend / alloc_decode, using the per-pool slot
indices returned in ``DSV4OutCacheLoc``.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class DSV4NPUReqToTokenPool(ReqToTokenPool):
    """ReqToTokenPool extended with DSV4 SWA + c4/c128 per-req tables.

    Drop-in replacement for ReqToTokenPool when the model is DeepSeek-V4 on
    NPU. Selected by ``model_runner_kv_cache_mixin`` based on model arch +
    device. Non-DSV4 and non-NPU paths continue to use the base class.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        super().__init__(size, max_context_len, device, enable_memory_saver)

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        # max_context_len is the max kv sequence length per request. Each
        # compressed pool stores 1 token per `ratio` raw tokens, so its
        # per-req table column count divides by ratio. State pools mirror
        # the raw token count (one state entry per raw position).
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token_swa = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )
            self.req_to_token_c4 = torch.zeros(
                (self._alloc_size, max(1, max_context_len // 4)),
                dtype=torch.int32,
                device=device,
            )
            self.req_to_token_c128 = torch.zeros(
                (self._alloc_size, max(1, max_context_len // 128)),
                dtype=torch.int32,
                device=device,
            )
            self.req_to_token_c4_state = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )
            self.req_to_token_c128_state = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

    # ------------------------------------------------------------------
    # Per-pool write helpers. mem_cache/common.py calls these directly
    # after a successful alloc_extend/alloc_decode, using slot indices
    # pulled from DSV4OutCacheLoc. Each method accepts (req_pool_idx,
    # token_offset) coordinates and the corresponding flat slot id.
    # ------------------------------------------------------------------

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

    def clear(self):
        super().clear()
        # No need to zero the auxiliary tables: they're indexed only by
        # active rows (via req_pool_idx) and only the [:seq_len] prefix
        # of each row is read. Stale entries past kv_committed_len are
        # unreachable by the attention metadata builder.
