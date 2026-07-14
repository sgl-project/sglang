from typing import List, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs


class SchedulerRecvSkipper:
    @staticmethod
    def maybe_create(server_args: ServerArgs):
        if server_args.scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper(server_args)

    @staticmethod
    def derive_forward_mode(gathered_modes: List[int]) -> Optional[ForwardMode]:
        """Collapse the gathered per-DP-rank forward modes into one weight-table
        bucket; the input is rank-identical, so the recv decision is too."""
        active = set(gathered_modes) - {
            ForwardMode.IDLE.value,
            ForwardMode.PREBUILT.value,
        }
        if not active:
            return None  # globally idle: same bucket as "no last batch"
        if active - {ForwardMode.DECODE.value, ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.EXTEND  # any extend-like rank: prompt recv
        if active == {ForwardMode.TARGET_VERIFY.value}:
            return ForwardMode.TARGET_VERIFY
        return ForwardMode.DECODE

    def __init__(self, server_args: ServerArgs):
        # Safe under DP-attention: ``handle`` is fed a mode derived from the
        # gathered per-rank modes, so every rank makes the same recv decision.
        self._counter = 0
        self._threshold = server_args.scheduler_recv_interval
        # All can be tuned if needed
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def handle(self, last_forward_mode: ForwardMode):
        should_recv = False

        last_weight = self._weight_of_forward_mode.get(
            last_forward_mode, self._default_weight
        )
        self._counter += last_weight

        if self._counter >= self._threshold:
            self._counter = 0
            should_recv = True

        return should_recv
