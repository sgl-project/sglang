"""Forward-time auxiliary output for ``ModelRunner.forward_observer``.

Outputs are copied to the host with generation results and consumed before
the batch is streamed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.auxiliary_output import DeviceAuxiliaryOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ForwardObserver(Protocol):
    def after_forward(
        self,
        batch: ScheduleBatch,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        *,
        can_run_graph: bool,
    ) -> Optional[DeviceAuxiliaryOutput]:
        """Return device output for later scheduler-side copying, or ``None``.

        Runs on the scheduler thread and forward stream after forward launch.
        Snapshot needed ``batch`` fields here; it may change before consumption.
        With ``can_run_graph``, copy any retained logits-output views: a later
        replay can overwrite the underlying CUDA graph buffers.
        """
        ...
