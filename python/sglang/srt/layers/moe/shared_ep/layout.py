"""Byte layout and typed tensor views for SharedEP VMM storage."""

from __future__ import annotations

import math

import msgspec
import torch


class SharedEpInputViews(msgspec.Struct, kw_only=True):
    activations: torch.Tensor
    scales: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    def owner(self, owner: int) -> SharedEpInputViews:
        return SharedEpInputViews(
            activations=self.activations[owner],
            scales=self.scales[owner],
            topk_ids=self.topk_ids[owner],
            topk_weights=self.topk_weights[owner],
        )


class SharedEpLayout(msgspec.Struct, frozen=True, kw_only=True):
    hidden_size: int
    top_k: int
    max_tokens_per_rank: int
    scale_groups: int
    input_payload_bytes: int
    input_row_bytes: int
    output_payload_bytes: int
    output_row_bytes: int

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        top_k: int,
        max_tokens_per_rank: int,
    ) -> SharedEpLayout:
        for name, value in (
            ("hidden_size", hidden_size),
            ("top_k", top_k),
            ("max_tokens_per_rank", max_tokens_per_rank),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if hidden_size % 128 != 0:
            raise ValueError(
                f"hidden_size must be divisible by FP8 group size 128, got {hidden_size}"
            )

        scale_groups = hidden_size // 128
        input_payload_bytes = hidden_size + scale_groups * 4 + top_k * 4 * 2
        output_payload_bytes = hidden_size * 2
        return cls(
            hidden_size=hidden_size,
            top_k=top_k,
            max_tokens_per_rank=max_tokens_per_rank,
            scale_groups=scale_groups,
            input_payload_bytes=input_payload_bytes,
            input_row_bytes=max(
                64 * 1024,
                _next_power_of_two(input_payload_bytes),
            ),
            output_payload_bytes=output_payload_bytes,
            output_row_bytes=max(
                16 * 1024,
                _next_power_of_two(output_payload_bytes),
            ),
        )

    @property
    def input_rank_bytes(self) -> int:
        return self.max_tokens_per_rank * self.input_row_bytes

    @property
    def output_rows_per_rank(self) -> int:
        return self.max_tokens_per_rank * self.top_k

    @property
    def output_rank_bytes(self) -> int:
        return self.output_rows_per_rank * self.output_row_bytes

    @property
    def scale_offset(self) -> int:
        return self.hidden_size

    @property
    def topk_id_offset(self) -> int:
        return self.scale_offset + self.scale_groups * 4

    @property
    def topk_weight_offset(self) -> int:
        return self.topk_id_offset + self.top_k * 4

    def output_slot_offset(self, token: int, route: int) -> int:
        if not 0 <= token < self.max_tokens_per_rank:
            raise IndexError(f"token {token} is outside the owner-local capacity")
        if not 0 <= route < self.top_k:
            raise IndexError(f"route {route} is outside Top-K {self.top_k}")
        return (token * self.top_k + route) * self.output_row_bytes

    def input_views(
        self,
        storage: torch.Tensor,
        *,
        world_size: int,
        mapped_rank_bytes: int,
    ) -> SharedEpInputViews:
        _validate_storage(
            storage,
            world_size=world_size,
            mapped_rank_bytes=mapped_rank_bytes,
            logical_rank_bytes=self.input_rank_bytes,
        )
        rows = torch.as_strided(
            storage,
            size=(world_size, self.max_tokens_per_rank, self.input_row_bytes),
            stride=(mapped_rank_bytes, self.input_row_bytes, 1),
        )
        scale_end = self.scale_offset + self.scale_groups * 4
        id_end = self.topk_id_offset + self.top_k * 4
        weight_end = self.topk_weight_offset + self.top_k * 4
        return SharedEpInputViews(
            activations=rows[..., : self.hidden_size].view(torch.float8_e4m3fn),
            scales=rows[..., self.scale_offset : scale_end].view(torch.float32),
            topk_ids=rows[..., self.topk_id_offset : id_end].view(torch.int32),
            topk_weights=rows[..., self.topk_weight_offset : weight_end].view(
                torch.float32
            ),
        )

    def output_view(
        self,
        storage: torch.Tensor,
        *,
        world_size: int,
        mapped_rank_bytes: int,
    ) -> torch.Tensor:
        _validate_storage(
            storage,
            world_size=world_size,
            mapped_rank_bytes=mapped_rank_bytes,
            logical_rank_bytes=self.output_rank_bytes,
        )
        rows = torch.as_strided(
            storage,
            size=(world_size, self.output_rows_per_rank, self.output_row_bytes),
            stride=(mapped_rank_bytes, self.output_row_bytes, 1),
        )
        values = rows[..., : self.output_payload_bytes].view(torch.bfloat16)
        return values.view(
            world_size,
            self.max_tokens_per_rank,
            self.top_k,
            self.hidden_size,
        )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def align_output_layout(
    layout: SharedEpLayout,
    *,
    granularity: int,
) -> SharedEpLayout:
    """Choose a fixed row stride so mapped owner segments have no hidden gap."""

    if granularity <= 0:
        raise ValueError(f"granularity must be positive, got {granularity}")
    rows = layout.output_rows_per_rank
    rank_alignment = math.lcm(rows, granularity)
    required_rank_bytes = rows * layout.output_row_bytes
    aligned_rank_bytes = (
        (required_rank_bytes + rank_alignment - 1) // rank_alignment
    ) * rank_alignment
    output_row_bytes = aligned_rank_bytes // rows
    if output_row_bytes == layout.output_row_bytes:
        return layout
    return SharedEpLayout(
        hidden_size=layout.hidden_size,
        top_k=layout.top_k,
        max_tokens_per_rank=layout.max_tokens_per_rank,
        scale_groups=layout.scale_groups,
        input_payload_bytes=layout.input_payload_bytes,
        input_row_bytes=layout.input_row_bytes,
        output_payload_bytes=layout.output_payload_bytes,
        output_row_bytes=output_row_bytes,
    )


def _validate_storage(
    storage: torch.Tensor,
    *,
    world_size: int,
    mapped_rank_bytes: int,
    logical_rank_bytes: int,
) -> None:
    if storage.dtype != torch.uint8 or storage.ndim != 1:
        raise TypeError("SharedEP VMM storage must be a one-dimensional uint8 tensor")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if mapped_rank_bytes < logical_rank_bytes:
        raise ValueError(
            "mapped rank bytes cannot be smaller than the logical rank payload"
        )
    required_bytes = (world_size - 1) * mapped_rank_bytes + logical_rank_bytes
    if storage.numel() < required_bytes:
        raise ValueError(
            f"SharedEP storage has {storage.numel()} bytes, needs {required_bytes}"
        )
