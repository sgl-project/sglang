"""Extension contract for forward-time auxiliary outputs.

Out-of-tree integrations install a ``ForwardObserver`` through
``ModelRunner.forward_observer``, from a model-runner subclass or a plugin
hook, to read what a target forward produced, e.g. its captured hidden states.
The observer's ``DeviceAuxiliaryOutput`` follows the lifecycle of
sampling-observer output: it is copied to the host with the generation result,
and its ``HostAuxiliaryOutput.consume`` runs before the batch is streamed.
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

        Runs on the scheduler thread right after the forward is launched, on
        the forward's stream, so work it enqueues is ordered after the forward.
        ``batch`` may change before the result is processed; read what is
        needed here. With ``can_run_graph`` the logits output can alias CUDA
        graph buffers that a later replay overwrites, so the returned output
        must not keep views of them.
        """
        ...
