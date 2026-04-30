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

    @property
    def shape(self):
        return self.kv_score.shape

    def __post_init__(self):
        self._item_size = self.kv_score.shape[-1] // 2

    @staticmethod
    def from_kv_score(*, kv: torch.Tensor, score: torch.Tensor) -> KVAndScore:
        assert kv.shape == score.shape
        return KVAndScore(torch.cat([kv, score], dim=-1))

    def new_empty(self, new_shape) -> KVAndScore:
        assert new_shape[-1] == self._item_size
        new_shape = list(new_shape)
        new_shape[-1] = 2 * self._item_size
        return KVAndScore(self.kv_score.new_empty(new_shape, requires_grad=False))

    def __getitem__(self, index) -> KVAndScore:
        return KVAndScore(self.kv_score[index])

    def __setitem__(self, index, value: KVAndScore):
        self.kv_score[index] = value.kv_score

    def clear(self):
        self.kv.zero_()
        self.score.fill_(float("-inf"))

    def view(self, *args):
        args = list(args)
        if isinstance(args[-1], int) and args[-1] != -1:
            args[-1] = 2 * self._item_size
        return KVAndScore(self.kv_score.view(*args))

    def clone(self) -> KVAndScore:
        return KVAndScore(self.kv_score.clone())

    @staticmethod
    def cat(tensors: list[KVAndScore], dim: int) -> KVAndScore:
        assert dim != -1, "Concatenation along last dim is not supported."
        assert len(tensors) > 0, "At least one tensor is required for concatenation."
        item_size = tensors[0]._item_size
        for v in tensors:
            assert (
                v._item_size == item_size
            ), "All tensors must have the same item size."

        return KVAndScore(torch.cat([v.kv_score for v in tensors], dim=dim))


class DeepSeekV4CompressState:
    def __init__(
        self,
        max_num_reqs: int,
        ratio: int,
        overlap: bool,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
        enable_memory_saver: bool = True,
    ):
        self.max_num_reqs = max_num_reqs
        self.ratio = ratio
        self.overlap = overlap
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        coff = 1 + self.overlap

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        state_shape = (max_num_reqs, ratio * coff, 2 * head_dim * coff)
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.kv_score_state = torch.empty(state_shape, dtype=dtype, device=device)

    def get_state(self) -> KVAndScore:
        return KVAndScore(self.kv_score_state)


class CompressStatePool:
    def __init__(
        self,
        size: int,
        swa_page_size: int,
        ring_size: int,
        overlap: bool,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        ratio: int,
        online: bool = False,
    ):
        self.swa_page_size = swa_page_size
        self.ring_size = ring_size
        self.enable_memory_saver = enable_memory_saver
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

    def translate_from_swa_loc_to_state_loc(
        self, swa_loc: torch.Tensor
    ) -> torch.Tensor:
        swa_pages = swa_loc // self.swa_page_size
        state_loc = swa_pages * self.ring_size + (swa_loc % self.ring_size)
        state_loc = torch.where(swa_loc < 0, -1, state_loc)
        return state_loc

    def get_state_by_state_loc(self, state_loc: torch.Tensor) -> KVAndScore:
        return self.kv_score_buffer[state_loc]

    def set_state_by_state_loc(self, state_loc: torch.Tensor, value: KVAndScore):
        self.kv_score_buffer[state_loc] = value
        self.kv_score_buffer[-1].clear()
