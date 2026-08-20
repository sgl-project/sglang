"""Extension contracts for sampling-time auxiliary response metadata.

Out-of-tree integrations can install these hooks from an SGLang plugin by
extending ``ModelRunner``, ``Scheduler``, and ``TokenizerManager`` through the
plugin hook registry. The model runner installs a ``SamplingObserver``; the
scheduler selects a ``SchedulerOutputStreamer`` subclass; and the tokenizer
manager consumes the resulting customized response fields.
"""

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


class SamplingObserver(Protocol):
    """Invocation-scoped hooks around the production grammar mask and sampler.

    Returning ``None`` from ``before_grammar`` skips ``after_sample``. Install
    an observer through ``ModelRunner.sampling_observer`` from a model-runner
    subclass or plugin hook. Specialized sampling paths must override
    ``ModelRunner.supports_sampling_observer`` and publish equivalent auxiliary
    output before installing one.
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
