from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from math import gcd

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


@dataclasses.dataclass
class KVAndScore:
    kv_score: torch.Tensor

    @property
    def kv(self) -> torch.Tensor:
        return self.kv_score[..., : self._item_size]

    @property
    def score(self) -> torch.Tensor:
        return self.kv_score[..., self._item_size :]

    def __post_init__(self):
        self._item_size = self.kv_score.shape[-1] // 2

    def __getitem__(self, index) -> KVAndScore:
        return KVAndScore(self.kv_score[index])

    def clear(self):
        self.kv.zero_()
        self.score.fill_(float("-inf"))


class CompressStatePool:
    def __init__(
        self,
        size: int,
        ring_size: int,
        overlap: bool,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        ratio: int,
        online: bool = False,
        page_size: int = 1,
    ):
        self.ring_size = ring_size
        self.online = online
        self.page_size = page_size

        if online:
            assert ring_size == 1, "online compress requires ring_size=1"
            self._size = size + self.ring_size + 1
            last_dim = 3 * head_dim
        else:
            self._size = size + self.ring_size + 1
            # Pad to lcm(ratio, page_size) so the flat buffer reshapes cleanly
            # into [block_num, page_size, last_dim] for the fused compressor
            # op (torch.ops.custom.compressor wants its state_cache in that
            # 3D layout). page_size=1 (default) falls back to the original
            # ratio-only padding.
            pad_to = _lcm(ratio, page_size) if page_size > 1 else ratio
            self._size = (self._size + pad_to - 1) // pad_to * pad_to
            last_dim = 2 * (1 + overlap) * head_dim

        self.last_dim = last_dim
        self._alloc_kv_score_buffer(
            dtype=dtype, device=device, enable_memory_saver=enable_memory_saver
        )
        if not online:
            self.kv_score_buffer[-1].clear()

    def _alloc_kv_score_buffer(
        self, *, dtype: torch.dtype, device: str, enable_memory_saver: bool
    ) -> None:
        """Allocate the flat ``(self._size, self.last_dim)`` kv+score buffer
        under the memory-saver / custom-mem-pool context and wrap it in
        :class:`KVAndScore`. Sets ``self.memory_saver_adapter``,
        ``self.custom_mem_pool`` and ``self.kv_score_buffer``.

        Subclasses (e.g. :class:`NPUCompressStatePool`) that compute a
        different ``self._size`` reuse this instead of duplicating the
        allocation boilerplate. Requires ``self._size`` and ``self.last_dim``
        to be set already.
        """
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=device)
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_score_buffer = KVAndScore(
                    torch.empty(
                        (self._size, self.last_dim),
                        dtype=dtype,
                        device=device,
                    )
                )

    @property
    def state_cache_3d(self) -> torch.Tensor:
        """``[block_num, page_size, last_dim]`` view of the flat kv+score
        buffer. ``last_dim = 2*(1+overlap)*head_dim`` — exactly the
        ``2*coff*D`` layout the fused compressor op wants for its
        ``state_cache`` argument (kv at ``[:, :, :coff*D]``, score at
        ``[:, :, coff*D:]``). Only valid for the non-online buffer; the
        online layout has ``last_dim = 3*head_dim`` which the fused path
        doesn't use.
        """
        assert not self.online, (
            "state_cache_3d is for the fused compressor path; "
            "online (3*head_dim) buffer is indexer-only."
        )
        assert self.page_size > 1, (
            "state_cache_3d requires page_size>1; pool was constructed "
            "with the default page_size=1 (flat 2D layout)."
        )
        return self.kv_score_buffer.kv_score.view(
            -1, self.page_size, self.last_dim
        )
