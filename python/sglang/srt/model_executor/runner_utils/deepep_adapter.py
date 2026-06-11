"""DeepEP capture/replay adapter — records the dispatch mode used during
capture and re-applies it during replay so DeepEP all-to-all has
consistent expert routing across the captured graph.
"""

from __future__ import annotations

from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_a2a_backend


class DeepEPCudaGraphRunnerAdapter:
    def __init__(self) -> None:
        # Record DeepEP mode used during capture to ensure replay consistency.
        self._captured_deepep_mode = None

    def capture(self, is_extend_in_batch: bool) -> None:
        if not get_moe_a2a_backend().is_deepep():
            return
        self._captured_deepep_mode = get_deepep_mode().resolve(
            is_extend_in_batch=is_extend_in_batch
        )
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)

    def replay(self) -> None:
        if not get_moe_a2a_backend().is_deepep():
            return
        assert self._captured_deepep_mode is not None
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)
