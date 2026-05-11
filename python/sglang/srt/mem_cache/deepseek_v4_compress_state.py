from __future__ import annotations

import dataclasses
from contextlib import nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
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


@dataclasses.dataclass
class KVAndScoreOld:
    """Legacy layout: kv and score stored as separate tensors of equal shape.
    (``SGLANG_CPU_COMPRESS_PATH=1``).
    """

    kv: torch.Tensor
    score: torch.Tensor

    def __post_init__(self):
        assert self.kv.shape == self.score.shape

    @property
    def shape(self):
        return self.kv.shape

    @staticmethod
    def empty_like(new_shape, old: "KVAndScoreOld") -> "KVAndScoreOld":
        return KVAndScoreOld(
            kv=old.kv.new_empty(new_shape),
            score=old.score.new_empty(new_shape),
        )

    def new_empty(self, new_shape) -> "KVAndScoreOld":
        return KVAndScoreOld.empty_like(new_shape, self)

    def __getitem__(self, index) -> "KVAndScoreOld":
        return KVAndScoreOld(kv=self.kv[index], score=self.score[index])

    def __setitem__(self, index, value: "KVAndScoreOld"):
        self.kv[index] = value.kv
        self.score[index] = value.score

    def view(self, *args) -> "KVAndScoreOld":
        return KVAndScoreOld(kv=self.kv.view(*args), score=self.score.view(*args))

    def clone(self) -> "KVAndScoreOld":
        return KVAndScoreOld(kv=self.kv.clone(), score=self.score.clone())

    def clear(self):
        self.kv.zero_()
        self.score.fill_(float("-inf"))


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

    def get_state(self):
        if envs.SGLANG_CPU_COMPRESS_PATH.get():
            half_dim = self.head_dim * (1 + self.overlap)
            return KVAndScoreOld(
                kv=self.kv_score_state[..., :half_dim],
                score=self.kv_score_state[..., half_dim:],
            )
        return KVAndScore(self.kv_score_state)


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
