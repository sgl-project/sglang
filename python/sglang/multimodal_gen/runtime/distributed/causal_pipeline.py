# SPDX-License-Identifier: Apache-2.0
"""Stage partitioning for same-timestep noisy history.

The schedule is independent of the model and transport. A rank owns a contiguous
piece of the expanded (denoising step, transformer layer) chain. It processes
chunks in order, satisfying both activation and same-stage history dependencies.
Timestep partitioning keeps entire forwards together; block partitioning can
split a forward across ranks.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CausalPipelinePlacement:
    """Physical ranks for denoising and the dedicated VAE decoder."""

    denoising_ranks: int

    def __post_init__(self):
        if self.denoising_ranks < 1:
            raise ValueError("Causal pipeline parallel size must be positive")

    @property
    def world_size(self) -> int:
        return self.denoising_ranks + 1


@dataclass(frozen=True)
class CausalPipelineSlice:
    timestep: int
    block_start: int
    block_end: int


@dataclass(frozen=True)
class CausalPipelinePlan:
    num_steps: int
    num_blocks: int
    degree: int
    mode: Literal["timestep", "block"]

    def __post_init__(self) -> None:
        if self.mode not in ("timestep", "block"):
            raise ValueError("Causal pipeline mode must be timestep or block")
        if min(self.num_steps, self.num_blocks, self.degree) < 1:
            raise ValueError("Causal pipeline dimensions must be positive")
        units = self.num_steps if self.mode == "timestep" else self.num_stages
        if self.degree > units:
            raise ValueError(
                f"{self.mode} partitioning has {units} units for {self.degree} ranks"
            )

    @property
    def num_stages(self) -> int:
        return self.num_steps * self.num_blocks

    def slices(self, rank: int) -> tuple[CausalPipelineSlice, ...]:
        if not 0 <= rank < self.degree:
            raise ValueError(f"Invalid causal pipeline rank {rank}")
        units = self.num_steps if self.mode == "timestep" else self.num_stages
        start = units * rank // self.degree
        end = units * (rank + 1) // self.degree
        if self.mode == "timestep":
            start *= self.num_blocks
            end *= self.num_blocks
        result = []
        while start < end:
            step, block_start = divmod(start, self.num_blocks)
            stop = min(end, (step + 1) * self.num_blocks)
            result.append(
                CausalPipelineSlice(step, block_start, stop - step * self.num_blocks)
            )
            start = stop
        return tuple(result)
