from __future__ import annotations

import dataclasses
from contextlib import nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils import is_hip
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

_is_hip = is_hip()


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
        swa_page_size: int = 0,
        online_mtp_max_draft_tokens: int = 0,
    ):
        self.ratio = ratio
        self.ring_size = ring_size
        self.swa_page_size = swa_page_size
        self.enable_memory_saver = enable_memory_saver
        self.online_mtp_state_slot_offset = 0
        self.online_mtp_max_draft_tokens = 0

        if online:
            assert ring_size == 1, "online compress requires ring_size=1"
            self._logical_size = size + self.ring_size + 1
            if online_mtp_max_draft_tokens > 0:
                # Bank 0 is the committed state. Banks 1..N cache per-draft
                # prefix states for lazy commit after target verify.
                self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
                self.online_mtp_state_slot_offset = self._logical_size
            self._size = self._logical_size * (1 + self.online_mtp_max_draft_tokens)
            last_dim = 3 * head_dim
        else:
            self._size = size + self.ring_size + 1
            self._size = (self._size + ratio - 1) // ratio * ratio
            self._logical_size = self._size
            last_dim = 2 * (1 + overlap) * head_dim

        if _is_hip:
            self.kv_score_buffer = KVAndScore(
                torch.empty((self._size, last_dim), dtype=dtype, device=device)
            )
            if not online:
                self.kv_score_buffer[-1].clear()
        else:
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

    def translate_from_req_position_to_state_loc(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        state_loc = req_pool_indices * self.ring_size + positions % self.ring_size
        state_loc = torch.where(positions < 0, -1, state_loc)
        return state_loc

    def get_state_by_state_loc(self, state_loc: torch.Tensor) -> KVAndScore:
        return self.kv_score_buffer[state_loc]

    def set_state_by_state_loc(self, state_loc: torch.Tensor, value: KVAndScore):
        self.kv_score_buffer[state_loc] = value
        self.kv_score_buffer[-1].clear()
