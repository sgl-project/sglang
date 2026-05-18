from __future__ import annotations

import dataclasses
from contextlib import nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


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
    ):
        self.ring_size = ring_size
        self.online = online

        if online:
            assert ring_size == 1, "online compress requires ring_size=1"
            self._size = size + self.ring_size + 1
            last_dim = 3 * head_dim
        else:
            self._size = size + self.ring_size + 1
            self._size = (self._size + ratio - 1) // ratio * ratio
            last_dim = 2 * (1 + overlap) * head_dim

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
                        (self._size, last_dim),
                        dtype=dtype,
                        device=device,
                    )
                )
                if not online:
                    self.kv_score_buffer[-1].clear()

    def get_cpu_copy(self, state_locs: torch.Tensor):
        if state_locs.numel() == 0:
            return None
        return self.kv_score_buffer.kv_score[state_locs].detach().to("cpu", copy=True)

    def load_cpu_copy(self, state_data, state_locs: torch.Tensor):
        if state_data is None or state_locs.numel() == 0:
            return
        device_data = state_data.to(
            self.kv_score_buffer.kv_score.device, non_blocking=True
        )
        self.kv_score_buffer.kv_score[state_locs] = device_data
        if not self.online:
            self.kv_score_buffer[-1].clear()
