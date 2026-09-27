"""Extension contracts for sampling-time auxiliary response metadata.

Out-of-tree integrations can install these hooks from an SGLang plugin by
extending ``ModelRunner``, ``Scheduler``, and ``TokenizerManager`` through the
plugin hook registry. The model runner installs a ``SamplingObserver``; the
scheduler selects a ``SchedulerOutputStreamer`` subclass; and the tokenizer
manager consumes the resulting customized response fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

import torch

from sglang.srt.managers.auxiliary_output import DeviceAuxiliaryOutput

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


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
