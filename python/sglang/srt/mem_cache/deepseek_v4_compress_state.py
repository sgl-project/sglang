from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from math import gcd

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils import is_hip, is_npu
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

_is_hip = is_hip()
_is_npu = is_npu()


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
            # Pad to lcm(ratio, page_size) so the flat buffer reshapes cleanly into
            # [block_num, page_size, last_dim] for the fused compressor op; page_size=1 falls back to ratio-only padding.
            pad_to = (
                _lcm(ratio, swa_page_size) if (swa_page_size > 1 and _is_npu) else ratio
            )
            self._size = (self._size + pad_to - 1) // pad_to * pad_to
            self._logical_size = self._size
            last_dim = 2 * (1 + overlap) * head_dim

        self.last_dim = last_dim
        self._alloc_kv_score_buffer(
            dtype=dtype, device=device, enable_memory_saver=enable_memory_saver
        )
        if not online:
            if _is_hip and ratio == 128:
                # Request-scoped C128 state is addressed by req_pool_idx (or a
                # per-request ring).  The pool is allocated with torch.empty(),
                # so a cold server can otherwise read uninitialized partial
                # states before a request slot has been written for the first
                # time.  Initialize all C128 rows to the empty-state sentinel;
                # C4 keeps the historical last-row sentinel behavior.
                self.kv_score_buffer.clear()
            else:
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
        if _is_hip:
            self.kv_score_buffer = KVAndScore(
                torch.empty((self._size, self.last_dim), dtype=dtype, device=device)
            )
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
        return self.kv_score_buffer.kv_score.view(-1, self.page_size, self.last_dim)

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


@dataclasses.dataclass
class KVAndScoreSeparate:
    """Legacy layout: kv and score stored as separate tensors of equal shape."""

    kv: torch.Tensor
    score: torch.Tensor

    def __post_init__(self):
        assert self.kv.shape == self.score.shape

    @property
    def shape(self):
        return self.kv.shape

    @staticmethod
    def empty_like(new_shape, sep: KVAndScoreSeparate) -> KVAndScoreSeparate:
        return KVAndScoreSeparate(
            kv=sep.kv.new_empty(new_shape),
            score=sep.score.new_empty(new_shape),
        )

    def new_empty(self, new_shape) -> KVAndScoreSeparate:
        return KVAndScoreSeparate.empty_like(new_shape, self)

    def __getitem__(self, index) -> KVAndScoreSeparate:
        return KVAndScoreSeparate(kv=self.kv[index], score=self.score[index])

    def __setitem__(self, index, value: KVAndScoreSeparate):
        self.kv[index] = value.kv
        self.score[index] = value.score

    def view(self, *args) -> KVAndScoreSeparate:
        return KVAndScoreSeparate(kv=self.kv.view(*args), score=self.score.view(*args))

    def clone(self) -> KVAndScoreSeparate:
        return KVAndScoreSeparate(kv=self.kv.clone(), score=self.score.clone())

    def clear(self):
        self.kv.zero_()
        self.score.fill_(float("-inf"))


class CompressStateSeparate:
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
        half_dim = self.head_dim * (1 + self.overlap)
        return KVAndScoreSeparate(
            kv=self.kv_score_state[..., :half_dim],
            score=self.kv_score_state[..., half_dim:],
        )
