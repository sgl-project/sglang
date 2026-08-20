from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


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
    """Scheduler-side result produced by ``DeviceAuxiliaryOutput``."""

    def consume(
        self,
        batch: ScheduleBatch,
        commits: Sequence[Optional[CommittedTokens]],
    ) -> None: ...


class SamplingObserver(Protocol):
    """Invocation-scoped hooks around the production grammar mask and sampler.

    Returning ``None`` from ``before_grammar`` skips ``after_sample``.
    These hooks are driven by ``ModelRunner.sample``. Specialized speculative
    workers that sample elsewhere must produce their own auxiliary output.
    """

    def is_active(self, sampling_info: SamplingBatchInfo) -> bool: ...

    def before_grammar(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
    ) -> Any: ...

    def after_sample(
        self, state: Any, token_ids: torch.Tensor
    ) -> Optional[DeviceAuxiliaryOutput]:
        """Return graph-safe device output for later scheduler-side copying."""
        ...
