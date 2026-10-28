"""Auxiliary outputs carried through the generation-result lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass(frozen=True)
class CommittedTokens:
    output_index: int
    token_ids: tuple[int, ...]


class DeviceAuxiliaryOutput(Protocol):
    """Device output copied later by the scheduler.

    Tensors must not alias CUDA-graph static buffers that a later replay can
    overwrite before the scheduler-side copy completes.
    """

    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> HostAuxiliaryOutput: ...


class HostAuxiliaryOutput(Protocol):
    """Scheduler-side result produced by ``DeviceAuxiliaryOutput``.

    ``consume`` runs after sampled tokens have been committed to each request
    and immediately before response streaming. ``commits`` is aligned with
    ``batch.reqs`` and identifies only the newly visible tokens. Implementations
    can buffer per-request values for a ``SchedulerOutputStreamer`` subclass to
    expose through customized response metadata.
    """

    def consume(
        self,
        batch: ScheduleBatch,
        commits: Sequence[Optional[CommittedTokens]],
    ) -> None: ...


@dataclass(frozen=True)
class CompositeHostAuxiliaryOutput:
    outputs: tuple[HostAuxiliaryOutput, ...]

    def consume(
        self,
        batch: ScheduleBatch,
        commits: Sequence[Optional[CommittedTokens]],
    ) -> None:
        for output in self.outputs:
            output.consume(batch, commits)


@dataclass(frozen=True)
class CompositeDeviceAuxiliaryOutput:
    outputs: tuple[DeviceAuxiliaryOutput, ...]

    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> CompositeHostAuxiliaryOutput:
        return CompositeHostAuxiliaryOutput(
            tuple(output.copy_to_host(copy_tensor) for output in self.outputs)
        )


def append_auxiliary_output(
    current: Optional[DeviceAuxiliaryOutput],
    output: Optional[DeviceAuxiliaryOutput],
) -> Optional[DeviceAuxiliaryOutput]:
    if output is None:
        return current
    if current is None:
        return output

    current_outputs = (
        current.outputs
        if isinstance(current, CompositeDeviceAuxiliaryOutput)
        else (current,)
    )
    new_outputs = (
        output.outputs
        if isinstance(output, CompositeDeviceAuxiliaryOutput)
        else (output,)
    )
    return CompositeDeviceAuxiliaryOutput(current_outputs + new_outputs)
