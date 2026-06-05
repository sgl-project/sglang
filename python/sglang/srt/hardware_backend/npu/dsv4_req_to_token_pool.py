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

        # Back-reference to the DSV4NPUTokenToKVPoolAllocator. Wired up via
        # ``register_dsv4_allocator`` after both are constructed (see
        # model_runner_kv_cache_mixin), so ``free(req)`` can release the
        # c4/c128 pool pages held by the req before the req_pool_idx slot
        # itself goes back to the free list. Stays None on construction so
        # the base class clear() in __init__ can run safely.
        self._dsv4_allocator = None

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
            # State tables: 1 slot per raw token (state pool is per-token in
            # the NPU paged layout used by torch.ops.custom.compressor
            # cache_mode=1). Backend builds
            # ``c{N}_state_page_table = req_to_token_c{N}_state[req, ::page_size]
            # // page_size`` to feed the kernel's state_block_table. Init zero
            # so unallocated columns map to block 0 (the kernel's skip
            # sentinel, see NPUCompressStatePool which clears block 0 to
            # kv=0/score=-inf).
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

    def register_dsv4_allocator(self, allocator) -> None:
        """Wire the DSV4NPUTokenToKVPoolAllocator back-ref so ``free(req)``
        can release c4/c128 pool pages alongside the req_pool_idx slot.

        Also wires the reverse direction: the allocator needs a back-ref
        to this pool so its c-pool alloc path can look up the previous
        compressed-token slot via ``get_last_loc`` on
        ``req_to_token_c{4,128}``. Without the reverse wiring the
        allocator falls back to a broken ``-1`` last_loc that opens a
        fresh c-pool page on every cross-boundary decode."""
        self._dsv4_allocator = allocator
        if hasattr(allocator, "register_req_to_token_pool"):
            allocator.register_req_to_token_pool(self)

    def free(self, req):
        # Trigger c4/c128 pool free via the allocator's unified free path;
        # the actual c-pool release logic lives in
        # ``DSV4NPUTokenToKVPoolAllocator.free``. _dsv4_allocator may be
        # None during the brief window between this pool's __init__ and
        # register_dsv4_allocator — defensive None check.
        if self._dsv4_allocator is not None:
            self._dsv4_allocator.free(req=req, req_to_token_pool=self)
        super().free(req)
