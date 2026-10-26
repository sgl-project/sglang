"""Torch/Triton dispatch pairs for DSpark kernels that only the parity tests call."""

from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    compact_row_index,
    compact_row_index_triton,
    compact_verify_ids,
    compact_verify_ids_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class CompactRowIndex:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        padded_total: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compact_row_index(
            verify_lens=verify_lens,
            padded_total=padded_total,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        padded_total: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compact_row_index_triton(
            verify_lens=verify_lens,
            padded_total=padded_total,
            device=device,
        )


class CompactVerifyIds:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        layout: RaggedVerifyLayout,
        device: str,
    ) -> torch.Tensor:
        return compact_verify_ids(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=device,
        )

    @classmethod
    def triton(
        cls,
        *,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        layout: RaggedVerifyLayout,
        device: str,
    ) -> torch.Tensor:
        return compact_verify_ids_triton(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=device,
        )
