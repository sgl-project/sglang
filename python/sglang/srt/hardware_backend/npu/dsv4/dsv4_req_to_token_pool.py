"""DSV4-NPU per-request mapping pool.

Subclass of ``ReqToTokenPool`` that adds five auxiliary per-request tables
needed by the DSV4 attention backend:

  * ``req_to_token_swa``        — slot ids in the SWA full-pool view
  * ``req_to_token_c4``         — slot ids in the c4 compressed-KV pool
  * ``req_to_token_c128``       — slot ids in the c128 compressed-KV pool
  * ``req_to_token_c4_state``   — c4 state-pool slot ids, 1 per raw token
  * ``req_to_token_c128_state`` — c128 state-pool slot ids, 1 per raw token

Compressed KV pools store 1 slot per ``ratio`` raw tokens, so their per-req
table column count is ``max_context_len // ratio``. swa mirrors the raw
token count. Elements are token-level slot ids; the attention backend
converts to page ids via ``// page_size`` when constructing PA_ND block
tables.

The c4/c128 STATE pools also have per-req tables here: the NPU fused
compressor uses a paged state pool (``cache_mode=1``), so each raw token's
state slot id is recorded (1 column per raw token) and the backend builds
``state_block_table = req_to_token_c{N}_state[req, ::page_size] // page_size``
to feed the kernel. (The base class' ``translate_kv_loc_to_compress_state_loc``
ring-hash is the CUDA-only path; it is disabled on NPU.)

Memory cost example (size=64, max_context_len=32K): swa 8MB + c4 2MB +
c128 64KB ≈ 10MB extra on top of the base req_to_token (8MB).

The tables are populated by the ``dsv4_common_hooks`` writers (driven from
``mem_cache/common.py``) immediately after a successful alloc_extend /
alloc_decode, using the per-pool slot indices returned in ``DSV4OutCacheLoc``.
"""

from __future__ import annotations

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class DSV4NPUReqToTokenPool(ReqToTokenPool):
    """ReqToTokenPool extended with DSV4 SWA + c4/c128 per-req tables.

    Drop-in replacement for ReqToTokenPool when the model is DeepSeek-V4 on
    NPU. Selected by ``model_runner_kv_cache_mixin`` based on model arch +
    device. Non-DSV4 and non-NPU paths continue to use the base class.

    The auxiliary tables are intentionally NOT zeroed on ``clear()``: they are
    indexed only by active rows (via req_pool_idx) and only each row's
    ``[:seq_len]`` prefix is read, so stale entries past kv_committed_len are
    unreachable by the attention metadata builder.
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

        # Back-ref to DSV4NPUTokenToKVPoolAllocator, wired via
        # register_dsv4_allocator after both exist, so free(req) can release
        # c4/c128 pages. None at construction so base clear() runs safely.
        self._dsv4_allocator = None

        # (name, columns). swa + state tables: 1 slot per raw token; c4/c128:
        # 1 slot per `ratio` raw tokens. Init zero so unallocated columns map to
        # block 0 (kernel skip sentinel cleared by NPUCompressStatePool).
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

    # ------------------------------------------------------------------
    # Per-pool write helpers, called by mem_cache/common.py after alloc, using
    # slot indices from DSV4OutCacheLoc. Args: (req_pool_idx, token_offset), slot.
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

    def register_dsv4_allocator(self, allocator) -> None:
        """Wire the DSV4NPUTokenToKVPoolAllocator ref so ``free(req)`` can
        release c4/c128 pool pages alongside the req_pool_idx slot. This is a
        one-way ref (pool -> allocator). The reverse direction (the allocator
        reading these per-req tables for its c-pool / state last_loc lookup) is
        no longer a stored back-ref: mem_cache/common.py passes this pool into
        ``alloc_extend`` / ``alloc_decode`` per call instead."""
        self._dsv4_allocator = allocator

    def free(self, req):
        # Trigger c4/c128 free via the allocator's unified free path. May be None
        # between __init__ and register_dsv4_allocator — defensive None check.
        if self._dsv4_allocator is not None:
            self._dsv4_allocator.free(req=req, req_to_token_pool=self)
        super().free(req)
