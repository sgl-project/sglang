"""Bounded physical layout for a windowed DFlash draft KV cache.

Absolute token positions remain the semantic anchor.  Every live request owns
three disjoint regions: a guard, a modulo-addressed committed window, and
stable verify scratch.  Generation checks prevent a recycled request slot from
silently inheriting an earlier owner's side cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError(
            f"invalid alignment geometry: value={value}, alignment={alignment}"
        )
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class CompactDFlashPhysicalLayout:
    owner_count: int
    window_size: int
    scratch_rows: int
    guard_rows: int
    page_size: int
    owner_span: int
    physical_tokens: int

    @classmethod
    def build(
        cls,
        *,
        owner_count: int,
        window_size: int,
        block_size: int,
        page_size: int,
        scratch_blocks: int = 2,
        guard_blocks: int = 2,
    ) -> CompactDFlashPhysicalLayout:
        values = {
            "owner_count": owner_count,
            "window_size": window_size,
            "block_size": block_size,
            "page_size": page_size,
            "scratch_blocks": scratch_blocks,
            "guard_blocks": guard_blocks,
        }
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError(f"compact DFlash layout values must be positive: {values}")
        if window_size % page_size:
            raise ValueError(
                f"window_size must be page aligned: window={window_size}, page={page_size}"
            )
        if window_size < block_size:
            raise ValueError(
                "window_size must cover one complete DFlash commit block: "
                f"window={window_size}, block={block_size}"
            )
        scratch_rows = _align_up(block_size * scratch_blocks, page_size)
        guard_rows = _align_up(block_size * guard_blocks, page_size)
        owner_span = _align_up(guard_rows + window_size + scratch_rows, page_size)
        return cls(
            owner_count=int(owner_count),
            window_size=int(window_size),
            scratch_rows=scratch_rows,
            guard_rows=guard_rows,
            page_size=int(page_size),
            owner_span=owner_span,
            physical_tokens=int(owner_count) * owner_span,
        )

    def _owner_bases(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        indices = req_pool_indices.to(torch.int64)
        if indices.numel() == 0:
            return indices
        valid = (indices >= 1) & (indices <= self.owner_count)
        if indices.is_cuda:
            torch._assert_async(
                torch.all(valid),
                "request slot is outside compact DFlash owner range",
            )
        elif not bool(torch.all(valid)):
            raise ValueError(
                "request slot is outside compact DFlash owner range: "
                f"min={int(indices.min())}, max={int(indices.max())}, "
                f"owner_count={self.owner_count}"
            )
        return (indices - 1) * self.owner_span

    def committed_locs(
        self, req_pool_indices: torch.Tensor, absolute_positions: torch.Tensor
    ) -> torch.Tensor:
        if req_pool_indices.shape != absolute_positions.shape:
            raise ValueError(
                "owner/position shape mismatch: "
                f"owners={tuple(req_pool_indices.shape)}, "
                f"positions={tuple(absolute_positions.shape)}"
            )
        positions = absolute_positions.to(torch.int64)
        if positions.numel():
            if positions.is_cuda:
                torch._assert_async(
                    torch.all(positions >= 0),
                    "absolute token positions must be nonnegative",
                )
            elif bool(torch.any(positions < 0)):
                raise ValueError("absolute token positions must be nonnegative")
        locs = (
            self._owner_bases(req_pool_indices)
            + self.guard_rows
            + torch.remainder(positions, self.window_size)
        )
        self.assert_in_bounds(locs)
        return locs

    def scratch_locs(self, req_pool_indices: torch.Tensor, width: int) -> torch.Tensor:
        if width <= 0 or width > self.scratch_rows:
            raise ValueError(
                f"scratch width outside reserved range: width={width}, "
                f"reserved={self.scratch_rows}"
            )
        bases = self._owner_bases(req_pool_indices.reshape(-1)).unsqueeze(1)
        offsets = torch.arange(width, device=bases.device, dtype=torch.int64).unsqueeze(
            0
        )
        locs = bases + self.guard_rows + self.window_size + offsets
        self.assert_in_bounds(locs)
        return locs

    def bind_first_use_or_assert_generation(
        self,
        req_pool_indices: torch.Tensor,
        owner_generation: torch.Tensor,
        current_generation: torch.Tensor,
        expected_generation: torch.Tensor,
        acquire_owner_mask: torch.Tensor,
    ) -> int:
        indices = req_pool_indices.to(torch.int64).reshape(-1).cpu()
        self._owner_bases(indices)
        if torch.unique(indices).numel() != indices.numel():
            raise RuntimeError("compact DFlash batch contains duplicate owner slots")
        expected = expected_generation.to(torch.int64).reshape(-1).cpu()
        acquire = acquire_owner_mask.to(torch.bool).reshape(-1).cpu()
        if expected.shape != indices.shape or acquire.shape != indices.shape:
            raise RuntimeError(
                "compact DFlash owner metadata shape mismatch: "
                f"indices={tuple(indices.shape)}, expected={tuple(expected.shape)}, "
                f"acquire={tuple(acquire.shape)}"
            )
        actual = current_generation[indices].to(torch.int64).cpu()
        if bool(torch.any(actual <= 0)) or bool(torch.any(expected <= 0)):
            raise RuntimeError(
                "compact DFlash owner has invalid allocation metadata: "
                f"expected={expected.tolist()}, actual={actual.tolist()}"
            )
        if not torch.equal(actual, expected):
            raise RuntimeError(
                "compact DFlash request generation mismatch: "
                f"expected={expected.tolist()}, actual={actual.tolist()}"
            )

        bound = owner_generation[indices].to(torch.int64).cpu()
        non_acquire_mismatch = (~acquire) & (bound != expected)
        if bool(torch.any(non_acquire_mismatch)):
            raise RuntimeError(
                "compact DFlash owner is not bound to the expected generation: "
                f"bound={bound.tolist()}, expected={expected.tolist()}, "
                f"acquire={acquire.tolist()}"
            )

        reused = acquire & (bound != 0) & (bound != expected)
        reuse_count = int(reused.sum().item())
        if bool(torch.any(acquire)):
            owner_generation[indices[acquire]] = expected[acquire]
        return reuse_count

    def assert_in_bounds(self, locs: torch.Tensor) -> None:
        if locs.numel() == 0:
            return
        valid = (locs >= 0) & (locs < self.physical_tokens)
        if locs.is_cuda:
            torch._assert_async(
                torch.all(valid),
                "compact DFlash physical location is out of bounds",
            )
            return
        minimum, maximum = int(locs.min()), int(locs.max())
        if minimum < 0 or maximum >= self.physical_tokens:
            raise RuntimeError(
                f"compact DFlash loc OOB: min={minimum}, max={maximum}, "
                f"physical_tokens={self.physical_tokens}"
            )
